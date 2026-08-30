"""
sql_store.py
Thin wrapper around the SQLite database plus lightweight entity extraction
(order IDs, customer IDs, customer names) so the router can turn a natural
language query into a safe, parameterized SQL lookup -- no LLM-generated
raw SQL is ever executed directly, which avoids SQL-injection-style risk
from unconstrained text-to-SQL.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import Config


@dataclass
class SQLResult:
    rows: List[dict] = field(default_factory=list)
    query_description: str = ""
    sql_used: str = ""

    def as_context_text(self) -> str:
        if not self.rows:
            return "No matching records were found in the database."
        lines = [self.query_description + ":"]
        for row in self.rows:
            lines.append(" - " + ", ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(lines)


class SQLStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db_path = cfg.paths.sqlite_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- Entity extraction helpers ----

    @staticmethod
    def extract_order_id(query: str) -> Optional[int]:
        match = re.search(r"\border\s*(?:#|number|no\.?)?\s*(\d{3,6})\b", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # fallback: any standalone 4-6 digit number likely an order id
        match = re.search(r"\b(\d{4,6})\b", query)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_customer_id(query: str) -> Optional[int]:
        match = re.search(r"\bcustomer\s*(?:#|id|number)?\s*(\d{1,4})\b", query, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def extract_customer_name(self, query: str) -> Optional[str]:
        """Match against known first/last names in the DB (simple exact-ish match)."""
        conn = self._connect()
        rows = conn.execute("SELECT DISTINCT first_name, last_name FROM customers").fetchall()
        conn.close()
        query_lower = query.lower()
        for row in rows:
            full_name = f"{row['first_name']} {row['last_name']}".lower()
            if full_name in query_lower or row["first_name"].lower() in query_lower:
                return full_name
        return None

    # ---- Query handlers ----

    def lookup_order(self, order_id: int) -> SQLResult:
        conn = self._connect()
        sql = """
            SELECT o.order_id, o.status, o.order_date, o.quantity,
                   p.name AS product, p.price,
                   c.first_name || ' ' || c.last_name AS customer
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_id = ?
        """
        rows = [dict(r) for r in conn.execute(sql, (order_id,)).fetchall()]
        conn.close()
        return SQLResult(
            rows=rows,
            query_description=f"Order lookup for order_id={order_id}",
            sql_used=sql.strip(),
        )

    def lookup_customer_orders(self, customer_id: Optional[int] = None,
                                customer_name: Optional[str] = None,
                                limit: int = 10) -> SQLResult:
        conn = self._connect()
        if customer_id is not None:
            sql = """
                SELECT o.order_id, o.status, o.order_date, o.quantity,
                       p.name AS product, p.price
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                WHERE o.customer_id = ?
                ORDER BY o.order_date DESC
                LIMIT ?
            """
            rows = [dict(r) for r in conn.execute(sql, (customer_id, limit)).fetchall()]
            desc = f"Orders for customer_id={customer_id}"
        elif customer_name is not None:
            sql = """
                SELECT o.order_id, o.status, o.order_date, o.quantity,
                       p.name AS product, p.price
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE LOWER(c.first_name || ' ' || c.last_name) = ?
                ORDER BY o.order_date DESC
                LIMIT ?
            """
            rows = [dict(r) for r in conn.execute(sql, (customer_name, limit)).fetchall()]
            desc = f"Orders for customer '{customer_name}'"
        else:
            rows, sql, desc = [], "", "No customer identifier found"
        conn.close()
        return SQLResult(rows=rows, query_description=desc, sql_used=sql.strip())

    def lookup_product(self, product_name_fragment: str) -> SQLResult:
        conn = self._connect()
        sql = "SELECT product_id, name, category, price FROM products WHERE LOWER(name) LIKE ?"
        rows = [dict(r) for r in conn.execute(sql, (f"%{product_name_fragment.lower()}%",)).fetchall()]
        conn.close()
        return SQLResult(
            rows=rows,
            query_description=f"Product lookup for '{product_name_fragment}'",
            sql_used=sql.strip(),
        )

    def lookup_orders_by_status(self, status: str, limit: int = 10) -> SQLResult:
        conn = self._connect()
        sql = """
            SELECT o.order_id, o.status, o.order_date, p.name AS product
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE LOWER(o.status) = ?
            ORDER BY o.order_date DESC
            LIMIT ?
        """
        rows = [dict(r) for r in conn.execute(sql, (status.lower(), limit)).fetchall()]
        conn.close()
        return SQLResult(
            rows=rows,
            query_description=f"Orders with status='{status}'",
            sql_used=sql.strip(),
        )

    def resolve_and_query(self, query: str) -> SQLResult:
        """
        Best-effort router: inspect the query text for known entity patterns
        and dispatch to the appropriate handler. This is intentionally simple
        (regex + keyword based) rather than free-form LLM-generated SQL, to
        keep execution safe and predictable.
        """
        order_id = self.extract_order_id(query)
        customer_id = self.extract_customer_id(query)
        customer_name = self.extract_customer_name(query)

        status_keywords = [s.lower() for s in ["Placed", "Shipped", "Delivered", "Cancelled", "Refunded"]]
        query_lower = query.lower()
        for status in status_keywords:
            if status in query_lower and "order" in query_lower:
                return self.lookup_orders_by_status(status, self.cfg.sql.max_rows_returned)

        if "order" in query_lower and order_id is not None and customer_id is None and customer_name is None:
            return self.lookup_order(order_id)

        if customer_id is not None:
            return self.lookup_customer_orders(customer_id=customer_id, limit=self.cfg.sql.max_rows_returned)

        if customer_name is not None:
            return self.lookup_customer_orders(customer_name=customer_name, limit=self.cfg.sql.max_rows_returned)

        if order_id is not None:
            return self.lookup_order(order_id)

        for _, product_name, _, _ in _known_product_names():
            if product_name.lower() in query_lower:
                return self.lookup_product(product_name)

        return SQLResult(rows=[], query_description="No structured entity recognized in query", sql_used="")


def _known_product_names():
    """Avoids importing the full PRODUCTS list to keep this module decoupled;
    duplicated here intentionally as a small, stable lookup table."""
    from src.generate_sample_data import PRODUCTS
    return PRODUCTS
