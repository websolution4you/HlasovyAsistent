import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from wallet_service import (
    BookingConflictError,
    InsufficientBalanceError,
    WalletRequiredError,
    booking_idempotency_key,
    create_ntc_booking_with_wallet,
    wallet_enabled_for_user,
)


class WalletServiceTests(unittest.TestCase):
    def test_wallet_feature_modes(self):
        allowed = {"user-1"}
        self.assertFalse(wallet_enabled_for_user("off", "user-1", allowed))
        self.assertFalse(wallet_enabled_for_user("invalid", "user-1", allowed))
        self.assertTrue(wallet_enabled_for_user("test", "USER-1", allowed))
        self.assertFalse(wallet_enabled_for_user("test", "user-2", allowed))
        self.assertTrue(wallet_enabled_for_user("on", "", allowed))

    def test_idempotency_key_is_stable_and_changes_with_operation(self):
        args = ("request-1", "user-1", "badminton-1", "badminton", "start", "end")
        first = booking_idempotency_key(*args)
        second = booking_idempotency_key(*args)
        changed = booking_idempotency_key("request-2", *args[1:])

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_rpc_result_is_parsed_as_decimal(self):
        rpc = Mock()
        rpc.execute.return_value = SimpleNamespace(
            data=[{
                "booking_id": "booking-1",
                "charged_eur": "10.00",
                "balance_eur": "15.50",
                "created": True,
            }]
        )
        supabase = Mock()
        supabase.rpc.return_value = rpc

        result = create_ntc_booking_with_wallet(
            supabase,
            user_id="user-1",
            court_id="badminton-1",
            sport="badminton",
            customer_name="Test User",
            customer_phone="+421900000000",
            start_at="2026-08-20T08:00:00+00:00",
            end_at="2026-08-20T09:00:00+00:00",
            notes="{}",
            idempotency_key="operation-1",
        )

        self.assertEqual(result.charged_eur, Decimal("10.00"))
        self.assertEqual(result.balance_eur, Decimal("15.50"))
        self.assertTrue(result.created)

    def test_unidentified_user_is_rejected_without_rpc(self):
        supabase = Mock()
        with self.assertRaises(WalletRequiredError):
            create_ntc_booking_with_wallet(
                supabase,
                user_id="",
                court_id="badminton-1",
                sport="badminton",
                customer_name="Guest",
                customer_phone="",
                start_at="start",
                end_at="end",
                notes="{}",
                idempotency_key="operation-1",
            )
        supabase.rpc.assert_not_called()

    def test_known_database_errors_are_mapped(self):
        cases = (
            ("Insufficient wallet balance", InsufficientBalanceError),
            ("Wallet does not exist for this NTC user", WalletRequiredError),
            ("The selected court is no longer available", BookingConflictError),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                rpc = Mock()
                rpc.execute.side_effect = RuntimeError(message)
                supabase = Mock()
                supabase.rpc.return_value = rpc
                with self.assertRaises(expected):
                    create_ntc_booking_with_wallet(
                        supabase,
                        user_id="user-1",
                        court_id="badminton-1",
                        sport="badminton",
                        customer_name="Test User",
                        customer_phone="+421900000000",
                        start_at="start",
                        end_at="end",
                        notes="{}",
                        idempotency_key="operation-1",
                    )


if __name__ == "__main__":
    unittest.main()
