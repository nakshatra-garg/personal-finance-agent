"""Budget tracking and expense management."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from config import AppConfig


@dataclass
class BudgetCategory:
    """Represents a budget category with limits."""
    name: str
    allocated: float
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        """Calculate remaining budget."""
        return self.allocated - self.spent

    @property
    def percent_used(self) -> float:
        """Calculate percentage of budget used."""
        if self.allocated == 0:
            return 0.0
        return (self.spent / self.allocated) * 100

    @property
    def status(self) -> str:
        """Get budget status."""
        percent = self.percent_used
        if percent >= 100:
            return "over"
        elif percent >= 80:
            return "warning"
        else:
            return "good"


class BudgetManager:
    """Manages budgets and tracks spending."""

    def __init__(self):
        """Initialize budget manager with default categories."""
        self.budgets: Dict[str, BudgetCategory] = {}
        self.transactions: List[Dict[str, Any]] = []

    def set_budget(self, category: str, amount: float):
        """
        Set budget for a category.

        Args:
            category: Budget category name
            amount: Budget amount
        """
        if category not in self.budgets:
            self.budgets[category] = BudgetCategory(name=category, allocated=amount)
        else:
            self.budgets[category].allocated = amount

    def add_transactions(self, transactions: List[Dict[str, Any]]):
        """
        Add transactions and update budgets.

        Args:
            transactions: List of transaction dictionaries
        """
        self.transactions.extend(transactions)
        self._update_budgets()

    def _update_budgets(self):
        """Update budget spending based on transactions."""
        # Reset spending
        for budget in self.budgets.values():
            budget.spent = 0.0

        # Calculate spending by category
        for txn in self.transactions:
            if txn["type"] == "expense":
                category = txn.get("category", "Other")
                amount = abs(txn["amount"])

                if category in self.budgets:
                    self.budgets[category].spent += amount

    def get_budget_summary(self) -> Dict[str, Any]:
        """
        Get overall budget summary.

        Returns:
            Summary with total allocated, spent, and status
        """
        total_allocated = sum(b.allocated for b in self.budgets.values())
        total_spent = sum(b.spent for b in self.budgets.values())

        categories = []
        for budget in self.budgets.values():
            categories.append({
                "name": budget.name,
                "allocated": budget.allocated,
                "spent": budget.spent,
                "remaining": budget.remaining,
                "percent_used": round(budget.percent_used, 1),
                "status": budget.status
            })

        # Sort by percent used (highest first)
        categories.sort(key=lambda x: x["percent_used"], reverse=True)

        return {
            "total_allocated": round(total_allocated, 2),
            "total_spent": round(total_spent, 2),
            "total_remaining": round(total_allocated - total_spent, 2),
            "categories": categories
        }

    def get_overspending_alerts(self) -> List[Dict[str, Any]]:
        """
        Get alerts for categories over budget.

        Returns:
            List of overspending alerts
        """
        alerts = []

        for budget in self.budgets.values():
            if budget.status == "over":
                alerts.append({
                    "category": budget.name,
                    "allocated": budget.allocated,
                    "spent": budget.spent,
                    "over_by": round(budget.spent - budget.allocated, 2),
                    "severity": "high"
                })
            elif budget.status == "warning":
                alerts.append({
                    "category": budget.name,
                    "allocated": budget.allocated,
                    "spent": budget.spent,
                    "percent_used": round(budget.percent_used, 1),
                    "severity": "medium"
                })

        return alerts

    def suggest_budget(
        self,
        monthly_income: float,
        savings_rate: float = 0.20
    ) -> Dict[str, float]:
        """
        Suggest budget allocation based on income and savings goal.

        Uses 50/30/20 rule as base:
        - 50% Needs (Housing, Food, Utilities, Transportation)
        - 30% Wants (Entertainment, Shopping, Dining out)
        - 20% Savings

        Args:
            monthly_income: Monthly income amount
            savings_rate: Desired savings rate (default 20%)

        Returns:
            Suggested budget by category
        """
        needs_percent = 0.50
        wants_percent = 0.30

        needs_budget = monthly_income * needs_percent
        wants_budget = monthly_income * wants_percent
        savings_budget = monthly_income * savings_rate

        # Allocate needs budget
        suggested = {
            "Housing": needs_budget * 0.40,  # 40% of needs
            "Food & Dining": needs_budget * 0.25,  # 25% of needs
            "Transportation": needs_budget * 0.20,  # 20% of needs
            "Utilities": needs_budget * 0.15,  # 15% of needs

            # Wants
            "Entertainment": wants_budget * 0.33,
            "Shopping": wants_budget * 0.33,
            "Personal Care": wants_budget * 0.17,
            "Subscriptions": wants_budget * 0.17,

            # Savings and other
            "Savings": savings_budget * 0.70,
            "Investments": savings_budget * 0.30,

            # Other categories
            "Healthcare": monthly_income * 0.05,
            "Education": monthly_income * 0.03,
            "Other": monthly_income * 0.02
        }

        # Round to 2 decimal places
        return {k: round(v, 2) for k, v in suggested.items()}

    def clear_transactions(self):
        """Clear all transactions and reset budgets."""
        self.transactions = []
        self._update_budgets()

    def export_budget_data(self) -> Dict[str, Any]:
        """
        Export budget and transaction data.

        Returns:
            Complete budget data for export
        """
        return {
            "budgets": [asdict(budget) for budget in self.budgets.values()],
            "transactions": self.transactions,
            "summary": self.get_budget_summary(),
            "export_date": datetime.now().isoformat()
        }
