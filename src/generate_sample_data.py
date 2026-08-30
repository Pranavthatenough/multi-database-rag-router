"""
generate_sample_data.py
Builds a synthetic e-commerce dataset with two distinct data sources, so the
router has a genuine reason to choose between them:

  1. A SQLite database (`company.db`) with customers, products, and orders --
     structured data best answered by SQL (e.g. "what's the status of order 1023").
  2. A folder of policy/FAQ text documents (`data/docs/`) -- unstructured
     knowledge best answered by similarity-based vector retrieval
     (e.g. "what's your return policy").

It also writes a labeled set of example queries (`data/labeled_queries.json`)
used to train and evaluate the query router classifier, and a small
`data/vector_eval.json` file with known relevant docs per query, used to
evaluate retrieval precision@k.
"""

import argparse
import json
import os
import random
import sqlite3


PRODUCTS = [
    (1, "Wireless Mouse", "Electronics", 19.99),
    (2, "Mechanical Keyboard", "Electronics", 79.99),
    (3, "USB-C Hub", "Electronics", 29.99),
    (4, "Running Shoes", "Footwear", 59.99),
    (5, "Yoga Mat", "Fitness", 24.99),
    (6, "Water Bottle", "Fitness", 14.99),
    (7, "Desk Lamp", "Home", 34.99),
    (8, "Office Chair", "Home", 149.99),
    (9, "Notebook Set", "Stationery", 9.99),
    (10, "Backpack", "Accessories", 44.99),
]

FIRST_NAMES = ["Alice", "Bob", "Carla", "David", "Emma", "Farhan", "Grace",
               "Hiro", "Isha", "Jacob", "Karen", "Liam", "Meera", "Noah"]
LAST_NAMES = ["Sharma", "Johnson", "Lee", "Martinez", "Patel", "Smith",
              "Chen", "Brown", "Nair", "Garcia", "Kim", "Ahmed"]

ORDER_STATUSES = ["Placed", "Shipped", "Delivered", "Cancelled", "Refunded"]

POLICY_DOCS = {
    "shipping_policy.txt": (
        "Shipping Policy\n\n"
        "We offer standard shipping (5-7 business days) and express shipping "
        "(2-3 business days) within the country. International shipping is "
        "available to over 40 countries and typically takes 10-15 business "
        "days. Orders over $50 qualify for free standard shipping. Shipping "
        "delays may occur during holiday seasons or due to customs "
        "processing for international orders. Once an order ships, "
        "customers receive a tracking link by email."
    ),
    "return_policy.txt": (
        "Return and Refund Policy\n\n"
        "Items can be returned within 30 days of delivery for a full refund, "
        "provided they are unused and in original packaging. To start a "
        "return, use the 'Return Item' button on your order history page. "
        "Refunds are processed within 5-7 business days after we receive the "
        "returned item. Sale items marked 'Final Sale' are not eligible for "
        "return. Refunds are issued to the original payment method."
    ),
    "warranty_policy.txt": (
        "Warranty Policy\n\n"
        "Electronics purchased from us come with a 1-year manufacturer "
        "warranty covering defects in materials and workmanship. Warranty "
        "does not cover accidental damage, water damage, or normal wear and "
        "tear. To file a warranty claim, contact support with your order "
        "number and a description of the defect. Approved claims are "
        "resolved via repair, replacement, or refund at our discretion."
    ),
    "account_help.txt": (
        "Account Help\n\n"
        "To reset your password, click 'Forgot Password' on the login page "
        "and follow the emailed link. To update your email address or "
        "shipping address, go to Account Settings. If you're locked out of "
        "your account after multiple failed login attempts, wait 15 minutes "
        "or contact support to unlock it immediately."
    ),
    "discount_policy.txt": (
        "Discounts and Promotions\n\n"
        "We run seasonal sales with discounts up to 40% off select "
        "categories. Newsletter subscribers receive a 10% off code for their "
        "first order. Discount codes cannot be combined with other "
        "promotions unless explicitly stated. Student and military "
        "discounts of 15% are available with valid verification through our "
        "partner service."
    ),
    "payment_faq.txt": (
        "Payment FAQ\n\n"
        "We accept major credit cards, debit cards, PayPal, and select "
        "digital wallets. Payments are processed securely and card details "
        "are never stored on our servers. If a payment fails, please check "
        "with your bank before retrying, as repeated failed attempts may "
        "temporarily flag your account for review."
    ),
}


def generate_sql_database(db_path: str, num_customers: int = 40, num_orders: int = 120, seed: int = 42) -> None:
    rng = random.Random(seed)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            email TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        )
    """)
    cur.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            status TEXT,
            order_date TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        )
    """)

    customers = []
    for cid in range(1, num_customers + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{cid}@example.com"
        customers.append((cid, first, last, email))
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)

    orders = []
    for oid in range(1001, 1001 + num_orders):
        customer_id = rng.randint(1, num_customers)
        product_id = rng.choice(PRODUCTS)[0]
        quantity = rng.randint(1, 3)
        status = rng.choice(ORDER_STATUSES)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"2026-{month:02d}-{day:02d}"
        orders.append((oid, customer_id, product_id, quantity, status, order_date))
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()
    print(f"SQLite DB written to '{db_path}': {num_customers} customers, "
          f"{len(PRODUCTS)} products, {num_orders} orders")


def generate_docs(docs_dir: str) -> None:
    os.makedirs(docs_dir, exist_ok=True)
    for filename, content in POLICY_DOCS.items():
        with open(os.path.join(docs_dir, filename), "w") as f:
            f.write(content)
    print(f"{len(POLICY_DOCS)} policy/FAQ documents written to '{docs_dir}'")


def generate_labeled_queries(data_dir: str) -> None:
    """
    Labeled examples for training/evaluating the router classifier.
    Labels: 'sql' (structured lookup), 'vector' (policy/FAQ knowledge),
    'hybrid' (needs both a specific record AND general policy knowledge).
    """
    examples = [
        # --- SQL: structured lookups ---
        ("What is the status of order 1023?", "sql"),
        ("Show me all orders placed by customer 5", "sql"),
        ("How many orders has Alice Sharma placed?", "sql"),
        ("What did customer 12 order last?", "sql"),
        ("List all cancelled orders", "sql"),
        ("What is the price of the Mechanical Keyboard?", "sql"),
        ("Which products are in the Electronics category?", "sql"),
        ("Give me the order date for order 1050", "sql"),
        ("How many products do we have in stock categories?", "sql"),
        ("What is the email address for customer 7?", "sql"),
        ("Show orders with status Delivered", "sql"),
        ("What quantity did order 1099 have?", "sql"),
        ("List the top 5 most recent orders", "sql"),
        ("Which customer placed order 1010?", "sql"),
        ("What products has customer 3 purchased?", "sql"),
        ("How many orders were shipped this year?", "sql"),
        ("What's the price of a Yoga Mat?", "sql"),
        ("Find the customer id for david.smith email", "sql"),
        ("What is order 1042's current status?", "sql"),
        ("Show me the order history for customer 20", "sql"),

        # --- Vector: general policy / FAQ knowledge ---
        ("What is your return policy?", "vector"),
        ("How long does shipping take?", "vector"),
        ("Do you ship internationally?", "vector"),
        ("How do I reset my password?", "vector"),
        ("What's covered under warranty?", "vector"),
        ("Can I combine discount codes?", "vector"),
        ("What payment methods do you accept?", "vector"),
        ("How do student discounts work?", "vector"),
        ("What happens if my payment fails?", "vector"),
        ("Is free shipping available?", "vector"),
        ("How do I update my shipping address?", "vector"),
        ("What items are not eligible for returns?", "vector"),
        ("How long is the electronics warranty?", "vector"),
        ("What if I'm locked out of my account?", "vector"),
        ("Do you offer a newsletter discount?", "vector"),
        ("How do refunds get issued?", "vector"),
        ("What's your policy on final sale items?", "vector"),
        ("How do I file a warranty claim?", "vector"),
        ("What happens during holiday shipping delays?", "vector"),
        ("Are card details stored on your servers?", "vector"),

        # --- Hybrid: needs a specific record AND policy context ---
        ("Can I return order 1023 given your return policy?", "hybrid"),
        ("Why is order 1042 delayed according to shipping policy?", "hybrid"),
        ("Is customer 5's last order still eligible for a refund?", "hybrid"),
        ("Does order 1010's product qualify for warranty coverage?", "hybrid"),
        ("Can order 1099 be cancelled based on our cancellation rules?", "hybrid"),
        ("Is the Mechanical Keyboard from order 1050 still under warranty?", "hybrid"),
        ("Should customer 12's delayed order get a shipping refund?", "hybrid"),
        ("Can Alice Sharma's most recent order still be returned?", "hybrid"),
        ("Was order 1023 charged correctly per our payment policy?", "hybrid"),
        ("Does customer 20 qualify for a student discount on their next order?", "hybrid"),
    ]

    with open(os.path.join(data_dir, "labeled_queries.json"), "w") as f:
        json.dump(
            [{"query": q, "label": l} for q, l in examples],
            f, indent=2,
        )
    print(f"{len(examples)} labeled router-training queries written")


def generate_vector_eval(data_dir: str) -> None:
    """Known relevant doc per query, for computing retrieval precision@k."""
    eval_set = [
        {"query": "What is your return policy?", "relevant_doc": "return_policy.txt"},
        {"query": "How long does shipping take?", "relevant_doc": "shipping_policy.txt"},
        {"query": "How do I reset my password?", "relevant_doc": "account_help.txt"},
        {"query": "What's covered under warranty?", "relevant_doc": "warranty_policy.txt"},
        {"query": "Can I combine discount codes?", "relevant_doc": "discount_policy.txt"},
        {"query": "What payment methods do you accept?", "relevant_doc": "payment_faq.txt"},
        {"query": "Do you ship internationally?", "relevant_doc": "shipping_policy.txt"},
        {"query": "What items are not eligible for returns?", "relevant_doc": "return_policy.txt"},
    ]
    with open(os.path.join(data_dir, "vector_eval.json"), "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"{len(eval_set)} vector-retrieval eval examples written")


def generate_all(data_dir: str = "data", seed: int = 42) -> None:
    docs_dir = os.path.join(data_dir, "docs")
    db_path = os.path.join(data_dir, "company.db")

    generate_sql_database(db_path, seed=seed)
    generate_docs(docs_dir)
    generate_labeled_queries(data_dir)
    generate_vector_eval(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic RAG router data")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_all(data_dir=args.data_dir, seed=args.seed)
