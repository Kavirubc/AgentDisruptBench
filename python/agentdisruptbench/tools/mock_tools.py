"""
AgentDisruptBench — Deterministic Mock Tools
=============================================

File:        mock_tools.py
Purpose:     Complete, deterministic mock tool implementations for all four
             benchmark domains: Retail, Travel, Finance, and DevOps.
             Same inputs always produce the same outputs.  Uses seeded Faker
             for realistic-looking data.  Tools reference each other
             consistently (e.g. product IDs from search_products are valid
             in check_inventory and place_order).

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    RetailTools  : search_products, check_inventory, place_order,
                   get_order_status, process_refund, get_customer_profile,
                   apply_coupon, update_cart.
    TravelTools  : search_flights, get_flight_details, book_flight,
                   cancel_booking, search_hotels, check_hotel_availability,
                   get_weather, currency_convert.
    FinanceTools : get_account_balance, transfer_funds,
                   get_transaction_history, get_exchange_rate,
                   validate_card, check_credit_limit.
    DevopsTools  : get_service_health, deploy_service, rollback_deployment,
                   get_logs, get_metrics, run_tests, create_incident,
                   resolve_incident.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger("agentdisruptbench.mock_tools")


# ---------------------------------------------------------------------------
# Helpers — deterministic data generation without external Faker dependency
# ---------------------------------------------------------------------------

def _deterministic_hash(seed: str) -> int:
    """Produce a stable integer from a string seed."""
    return int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)


def _det_id(prefix: str, seed: str) -> str:
    """Deterministic short ID like 'PRD-a3f8'."""
    h = hashlib.sha256(seed.encode()).hexdigest()[:6]
    return f"{prefix}-{h}"


def _det_price(seed: str, lo: float = 5.0, hi: float = 500.0) -> float:
    """Deterministic price in [lo, hi]."""
    v = _deterministic_hash(seed)
    return round(lo + (v % 10000) / 10000 * (hi - lo), 2)


def _det_name(seed: str, names: list[str]) -> str:
    """Pick a name deterministically."""
    return names[_deterministic_hash(seed) % len(names)]


# ---------------------------------------------------------------------------
# Retail Tools
# ---------------------------------------------------------------------------

_PRODUCT_NAMES = [
    "Blue Widget", "Red Gadget", "Green Sprocket", "Silver Bolt",
    "Gold Bracket", "Titanium Gear", "Carbon Fiber Panel",
    "LED Display Module", "Wireless Sensor Kit", "Smart Thermostat",
    "Precision Caliper", "Portable Battery Pack", "USB-C Hub",
    "Ergonomic Keyboard", "Noise-Cancelling Headphones",
]


class RetailTools:
    """Deterministic mock tools for the Retail domain.

    All methods are stateless — same inputs always produce the same outputs.
    Product IDs, customer IDs, and order IDs are internally consistent.
    """

    @staticmethod
    def search_products(*, query: str, max_results: int = 5) -> dict:
        """Search products by keyword.  Returns a list of matching products."""
        results = []
        for i in range(min(max_results, 5)):
            seed = f"product:{query}:{i}"
            results.append({
                "product_id": _det_id("PRD", seed),
                "name": _det_name(seed, _PRODUCT_NAMES),
                "price": _det_price(seed),
                "currency": "USD",
                "in_stock": (_deterministic_hash(seed) % 10) > 2,
                "rating": round(3.0 + (_deterministic_hash(seed + "r") % 20) / 10, 1),
            })
        return {"products": results, "total": len(results), "query": query}

    @staticmethod
    def check_inventory(*, product_id: str, warehouse: str = "default") -> dict:
        """Check inventory for a specific product."""
        seed = f"inventory:{product_id}:{warehouse}"
        qty = _deterministic_hash(seed) % 100
        return {
            "product_id": product_id,
            "warehouse": warehouse,
            "quantity_available": qty,
            "reserved": _deterministic_hash(seed + "res") % 10,
            "restock_date": "2026-04-01" if qty < 5 else None,
        }

    @staticmethod
    def place_order(
        *, customer_id: str, product_id: str, quantity: int = 1
    ) -> dict:
        """Place an order for a product."""
        seed = f"order:{customer_id}:{product_id}:{quantity}"
        order_id = _det_id("ORD", seed)
        unit_price = _det_price(f"product:{product_id}:0")
        return {
            "order_id": order_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": round(unit_price * quantity, 2),
            "status": "confirmed",
            "estimated_delivery": "2026-03-15",
        }

    @staticmethod
    def get_order_status(*, order_id: str) -> dict:
        """Get status of an existing order."""
        seed = f"status:{order_id}"
        statuses = ["processing", "shipped", "in_transit", "delivered"]
        return {
            "order_id": order_id,
            "status": _det_name(seed, statuses),
            "tracking_number": _det_id("TRK", seed),
            "last_updated": "2026-03-10T08:30:00Z",
        }

    @staticmethod
    def process_refund(*, order_id: str, reason: str = "customer_request") -> dict:
        """Process a refund for an order."""
        seed = f"refund:{order_id}"
        return {
            "refund_id": _det_id("RFN", seed),
            "order_id": order_id,
            "status": "approved",
            "amount": _det_price(seed, 10, 200),
            "reason": reason,
            "estimated_credit_date": "2026-03-12",
        }

    @staticmethod
    def get_customer_profile(*, customer_id: str) -> dict:
        """Retrieve customer profile."""
        seed = f"customer:{customer_id}"
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
        return {
            "customer_id": customer_id,
            "first_name": _det_name(seed + "fn", first_names),
            "last_name": _det_name(seed + "ln", last_names),
            "email": f"user{_deterministic_hash(seed) % 9999}@example.com",
            "tier": _det_name(seed + "tier", ["bronze", "silver", "gold", "platinum"]),
            "total_orders": _deterministic_hash(seed + "ord") % 50,
            "member_since": "2024-01-15",
        }

    @staticmethod
    def apply_coupon(*, cart_id: str, coupon_code: str) -> dict:
        """Apply a coupon to a cart."""
        seed = f"coupon:{cart_id}:{coupon_code}"
        discount_pct = (_deterministic_hash(seed) % 30) + 5
        valid = (_deterministic_hash(seed + "valid") % 3) != 0
        return {
            "cart_id": cart_id,
            "coupon_code": coupon_code,
            "valid": valid,
            "discount_percent": discount_pct if valid else 0,
            "message": "Coupon applied" if valid else "Invalid or expired coupon",
        }

    @staticmethod
    def update_cart(
        *, cart_id: str, product_id: str, quantity: int, action: str = "add"
    ) -> dict:
        """Add/remove items from a shopping cart."""
        seed = f"cart:{cart_id}:{product_id}:{action}"
        return {
            "cart_id": cart_id,
            "action": action,
            "product_id": product_id,
            "quantity": quantity,
            "cart_total": _det_price(seed, 20, 500),
            "item_count": (_deterministic_hash(seed) % 8) + 1,
        }


# ---------------------------------------------------------------------------
# Travel Tools
# ---------------------------------------------------------------------------

_AIRLINES = ["SkyWest Air", "Oceanic Airlines", "Pacific Wings", "Alpine Jet", "Metro Air"]
_HOTELS = ["Grand Plaza Hotel", "Seaside Resort", "Mountain Lodge", "City Central Inn", "Lakefront Suites"]


class TravelTools:
    """Deterministic mock tools for the Travel domain."""

    @staticmethod
    def search_flights(
        *, origin: str, destination: str, date: str, passengers: int = 1
    ) -> dict:
        """Search flights between two airports."""
        results = []
        for i in range(3):
            seed = f"flight:{origin}:{destination}:{date}:{i}"
            results.append({
                "flight_id": _det_id("FLT", seed),
                "airline": _det_name(seed, _AIRLINES),
                "origin": origin,
                "destination": destination,
                "departure": f"{date}T{8 + i * 4:02d}:00:00Z",
                "arrival": f"{date}T{11 + i * 4:02d}:30:00Z",
                "price_per_person": _det_price(seed, 150, 1200),
                "seats_available": _deterministic_hash(seed + "seats") % 50 + 1,
                "class": _det_name(seed + "cls", ["economy", "business", "first"]),
            })
        return {"flights": results, "total": len(results)}

    @staticmethod
    def get_flight_details(*, flight_id: str) -> dict:
        """Get detailed info for a specific flight."""
        seed = f"flightdetail:{flight_id}"
        return {
            "flight_id": flight_id,
            "airline": _det_name(seed, _AIRLINES),
            "aircraft": _det_name(seed + "ac", ["Boeing 737", "Airbus A320", "Boeing 787"]),
            "duration_minutes": 120 + _deterministic_hash(seed) % 360,
            "stops": _deterministic_hash(seed + "stops") % 3,
            "on_time_percentage": 75 + _deterministic_hash(seed + "otp") % 25,
            "baggage_allowance_kg": 23,
            "meal_included": (_deterministic_hash(seed + "meal") % 2) == 0,
        }

    @staticmethod
    def book_flight(
        *, flight_id: str, passenger_name: str, passenger_count: int = 1
    ) -> dict:
        """Book a flight."""
        seed = f"booking:{flight_id}:{passenger_name}"
        return {
            "booking_id": _det_id("BKG", seed),
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "passenger_count": passenger_count,
            "total_price": _det_price(seed, 200, 2000),
            "status": "confirmed",
            "confirmation_code": _det_id("CNF", seed).upper(),
        }

    @staticmethod
    def cancel_booking(*, booking_id: str, reason: str = "schedule_change") -> dict:
        """Cancel a flight booking."""
        seed = f"cancel:{booking_id}"
        return {
            "booking_id": booking_id,
            "status": "cancelled",
            "refund_amount": _det_price(seed, 50, 1000),
            "reason": reason,
            "cancellation_fee": _det_price(seed + "fee", 0, 100),
        }

    @staticmethod
    def search_hotels(
        *, location: str, check_in: str, check_out: str, guests: int = 1
    ) -> dict:
        """Search hotels at a location."""
        results = []
        for i in range(4):
            seed = f"hotel:{location}:{check_in}:{i}"
            results.append({
                "hotel_id": _det_id("HTL", seed),
                "name": _det_name(seed, _HOTELS),
                "location": location,
                "price_per_night": _det_price(seed, 80, 500),
                "rating": round(3.5 + (_deterministic_hash(seed + "r") % 15) / 10, 1),
                "availability": (_deterministic_hash(seed + "avl") % 5) > 0,
            })
        return {"hotels": results, "total": len(results)}

    @staticmethod
    def check_hotel_availability(
        *, hotel_id: str, check_in: str, check_out: str, room_type: str = "standard"
    ) -> dict:
        """Check room availability at a specific hotel."""
        seed = f"hotelavail:{hotel_id}:{check_in}:{room_type}"
        return {
            "hotel_id": hotel_id,
            "room_type": room_type,
            "available": (_deterministic_hash(seed) % 4) > 0,
            "rooms_left": _deterministic_hash(seed + "left") % 10,
            "price_per_night": _det_price(seed),
            "amenities": ["wifi", "breakfast", "pool"][:(_deterministic_hash(seed + "am") % 3 + 1)],
        }

    @staticmethod
    def get_weather(*, location: str, date: str) -> dict:
        """Get weather forecast for a location and date."""
        seed = f"weather:{location}:{date}"
        conditions = ["sunny", "cloudy", "rainy", "partly_cloudy", "stormy", "snowy"]
        return {
            "location": location,
            "date": date,
            "condition": _det_name(seed, conditions),
            "temperature_celsius": 5 + _deterministic_hash(seed) % 30,
            "humidity_percent": 30 + _deterministic_hash(seed + "hum") % 60,
            "wind_speed_kmh": _deterministic_hash(seed + "wind") % 40,
        }

    @staticmethod
    def currency_convert(
        *, amount: float, from_currency: str, to_currency: str
    ) -> dict:
        """Convert amount between currencies."""
        seed = f"fx:{from_currency}:{to_currency}"
        rate = 0.5 + (_deterministic_hash(seed) % 300) / 100
        return {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "amount": amount,
            "rate": round(rate, 4),
            "converted_amount": round(amount * rate, 2),
            "timestamp": "2026-03-09T12:00:00Z",
        }


# ---------------------------------------------------------------------------
# Finance Tools
# ---------------------------------------------------------------------------


class FinanceTools:
    """Deterministic mock tools for the Finance domain."""

    @staticmethod
    def get_account_balance(*, account_id: str) -> dict:
        """Get account balance."""
        seed = f"balance:{account_id}"
        return {
            "account_id": account_id,
            "balance": _det_price(seed, 100, 50000),
            "currency": "USD",
            "available_balance": _det_price(seed + "avl", 100, 45000),
            "pending_transactions": _deterministic_hash(seed + "pend") % 5,
            "last_updated": "2026-03-09T11:00:00Z",
        }

    @staticmethod
    def transfer_funds(
        *, from_account: str, to_account: str, amount: float, currency: str = "USD"
    ) -> dict:
        """Transfer funds between accounts."""
        seed = f"transfer:{from_account}:{to_account}:{amount}"
        return {
            "transfer_id": _det_id("TXF", seed),
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "currency": currency,
            "status": "completed",
            "fee": round(amount * 0.001, 2),
            "timestamp": "2026-03-09T11:05:00Z",
        }

    @staticmethod
    def get_transaction_history(
        *, account_id: str, days: int = 30, limit: int = 10
    ) -> dict:
        """Get recent transaction history."""
        txns = []
        for i in range(min(limit, 10)):
            seed = f"txn:{account_id}:{i}"
            txns.append({
                "transaction_id": _det_id("TXN", seed),
                "type": _det_name(seed, ["debit", "credit", "transfer", "payment"]),
                "amount": _det_price(seed, 5, 2000),
                "description": _det_name(
                    seed + "desc",
                    ["Grocery Store", "Online Purchase", "Salary Deposit",
                     "Utility Bill", "Restaurant", "ATM Withdrawal"],
                ),
                "date": f"2026-03-{max(1, 9 - i):02d}",
                "balance_after": _det_price(seed + "bal", 500, 30000),
            })
        return {"account_id": account_id, "transactions": txns, "total": len(txns)}

    @staticmethod
    def get_exchange_rate(*, base_currency: str, target_currency: str) -> dict:
        """Get current exchange rate."""
        seed = f"rate:{base_currency}:{target_currency}"
        rate = 0.5 + (_deterministic_hash(seed) % 300) / 100
        return {
            "base_currency": base_currency,
            "target_currency": target_currency,
            "rate": round(rate, 6),
            "inverse_rate": round(1 / rate, 6),
            "timestamp": "2026-03-09T12:00:00Z",
            "source": "ECB",
        }

    @staticmethod
    def validate_card(*, card_number: str, expiry: str, cvv: str) -> dict:
        """Validate a payment card."""
        seed = f"card:{card_number}"
        is_valid = (_deterministic_hash(seed) % 5) > 0  # 80% valid
        return {
            "card_number_masked": f"****{card_number[-4:]}",
            "valid": is_valid,
            "card_type": _det_name(seed, ["visa", "mastercard", "amex"]),
            "issuing_bank": _det_name(seed + "bank", ["Chase", "Citi", "BoA", "HSBC"]),
            "message": "Card validated" if is_valid else "Card declined",
        }

    @staticmethod
    def check_credit_limit(*, account_id: str) -> dict:
        """Check credit limit for an account."""
        seed = f"credit:{account_id}"
        limit = _det_price(seed, 1000, 50000)
        used = _det_price(seed + "used", 0, limit)
        return {
            "account_id": account_id,
            "credit_limit": limit,
            "used": round(used, 2),
            "available": round(limit - used, 2),
            "utilization_percent": round((used / limit) * 100, 1),
        }


# ---------------------------------------------------------------------------
# DevOps Tools
# ---------------------------------------------------------------------------

_SERVICE_NAMES = ["api-gateway", "auth-service", "payment-service", "user-service", "notification-service"]


class DevopsTools:
    """Deterministic mock tools for the DevOps domain."""

    @staticmethod
    def get_service_health(*, service_name: str) -> dict:
        """Get health status of a service."""
        seed = f"health:{service_name}"
        statuses = ["healthy", "degraded", "unhealthy"]
        return {
            "service": service_name,
            "status": _det_name(seed, statuses),
            "uptime_percent": round(95 + (_deterministic_hash(seed) % 50) / 10, 2),
            "response_time_ms": 20 + _deterministic_hash(seed + "rt") % 200,
            "active_connections": _deterministic_hash(seed + "conn") % 500,
            "last_checked": "2026-03-09T12:00:00Z",
        }

    @staticmethod
    def deploy_service(
        *, service_name: str, version: str, environment: str = "staging"
    ) -> dict:
        """Deploy a new version of a service."""
        seed = f"deploy:{service_name}:{version}:{environment}"
        return {
            "deployment_id": _det_id("DEP", seed),
            "service": service_name,
            "version": version,
            "environment": environment,
            "status": "success",
            "instances_updated": 3 + _deterministic_hash(seed) % 5,
            "rollback_version": f"v{_deterministic_hash(seed + 'prev') % 20}.{_deterministic_hash(seed + 'minor') % 10}.0",
            "timestamp": "2026-03-09T12:05:00Z",
        }

    @staticmethod
    def rollback_deployment(*, deployment_id: str, reason: str = "performance_regression") -> dict:
        """Rollback a deployment."""
        seed = f"rollback:{deployment_id}"
        return {
            "rollback_id": _det_id("RBK", seed),
            "deployment_id": deployment_id,
            "status": "completed",
            "rolled_back_to": f"v{_deterministic_hash(seed) % 20}.{_deterministic_hash(seed + 'm') % 10}.0",
            "reason": reason,
            "duration_seconds": 15 + _deterministic_hash(seed + "dur") % 60,
        }

    @staticmethod
    def get_logs(
        *, service_name: str, severity: str = "error", limit: int = 5
    ) -> dict:
        """Get recent logs for a service."""
        log_msgs = [
            "Connection timeout to database",
            "Rate limit exceeded for API endpoint /v1/users",
            "Memory usage above 90% threshold",
            "Failed to authenticate request — token expired",
            "Disk I/O latency spike detected on node-3",
            "Health check failed for downstream dependency",
            "Certificate renewal completed successfully",
            "Graceful shutdown initiated",
        ]
        logs = []
        for i in range(min(limit, 8)):
            seed = f"log:{service_name}:{i}"
            logs.append({
                "log_id": _det_id("LOG", seed),
                "severity": severity,
                "message": log_msgs[_deterministic_hash(seed) % len(log_msgs)],
                "timestamp": f"2026-03-09T{11 - i:02d}:30:00Z",
                "source": service_name,
            })
        return {"service": service_name, "logs": logs, "total": len(logs)}

    @staticmethod
    def get_metrics(
        *, service_name: str, metric_type: str = "cpu", period_minutes: int = 60
    ) -> dict:
        """Get performance metrics for a service."""
        seed = f"metrics:{service_name}:{metric_type}"
        data_points = []
        for i in range(6):
            s = f"{seed}:{i}"
            data_points.append({
                "timestamp": f"2026-03-09T{11 + i}:00:00Z",
                "value": round(10 + (_deterministic_hash(s) % 800) / 10, 2),
            })
        return {
            "service": service_name,
            "metric_type": metric_type,
            "unit": "percent" if metric_type in ("cpu", "memory") else "ms",
            "data_points": data_points,
            "average": round(sum(d["value"] for d in data_points) / len(data_points), 2),
            "peak": max(d["value"] for d in data_points),
        }

    @staticmethod
    def run_tests(*, service_name: str, test_suite: str = "unit") -> dict:
        """Run test suite for a service."""
        seed = f"tests:{service_name}:{test_suite}"
        total = 50 + _deterministic_hash(seed) % 150
        passed = total - _deterministic_hash(seed + "fail") % 5
        return {
            "service": service_name,
            "test_suite": test_suite,
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "skipped": _deterministic_hash(seed + "skip") % 3,
            "duration_seconds": round(5 + (_deterministic_hash(seed + "dur") % 600) / 10, 1),
            "coverage_percent": round(70 + (_deterministic_hash(seed + "cov") % 300) / 10, 1),
        }

    @staticmethod
    def create_incident(
        *, title: str, severity: str, service_name: str, description: str = ""
    ) -> dict:
        """Create an incident."""
        seed = f"incident:{title}:{service_name}"
        return {
            "incident_id": _det_id("INC", seed),
            "title": title,
            "severity": severity,
            "service": service_name,
            "status": "open",
            "assigned_to": _det_name(seed, ["oncall-team-alpha", "oncall-team-beta", "platform-eng"]),
            "created_at": "2026-03-09T12:10:00Z",
        }

    @staticmethod
    def resolve_incident(*, incident_id: str, resolution: str) -> dict:
        """Resolve an incident."""
        seed = f"resolve:{incident_id}"
        return {
            "incident_id": incident_id,
            "status": "resolved",
            "resolution": resolution,
            "resolved_at": "2026-03-09T13:00:00Z",
            "duration_minutes": 20 + _deterministic_hash(seed) % 180,
        }


# ---------------------------------------------------------------------------
# All tools registry helper
# ---------------------------------------------------------------------------

def get_all_tools() -> dict[str, Any]:
    """Return a flat dict mapping tool_name → callable for all domains.

    This is the convenience entry point used by the ToolRegistry and
    the BenchmarkRunner to obtain the complete tool suite.
    """
    tools: dict[str, Any] = {}
    for cls in (RetailTools, TravelTools, FinanceTools, DevopsTools):
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
            method = getattr(cls, attr_name)
            if callable(method):
                tools[attr_name] = method
    return tools
