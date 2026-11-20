from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database.models import Base, Customer, Membership, Tenant, User
from backend.database.session import build_engine


def test_sqlalchemy_tenant_customer_persistence_and_unique_constraint():
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    tenant_id, user_id = str(uuid4()), str(uuid4())
    with Session(engine) as db:
        db.add(Tenant(id=tenant_id, name="Test Org"))
        db.add(User(id=user_id, email="owner@example.test", password_hash="hashed", full_name="Owner"))
        db.flush()
        db.add(Membership(id=str(uuid4()), tenant_id=tenant_id, user_id=user_id, role="owner"))
        db.add(Customer(id=str(uuid4()), tenant_id=tenant_id, customer_id="C-1", features={"age": 40}))
        db.commit()
    with Session(engine) as db:
        customer = db.scalar(select(Customer).where(Customer.tenant_id == tenant_id))
        assert customer.customer_id == "C-1"
        assert customer.features == {"age": 40}
        db.add(Customer(id=str(uuid4()), tenant_id=tenant_id, customer_id="C-1", features={"age": 41}))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
        else:
            raise AssertionError("Duplicate tenant customer ID was accepted")
