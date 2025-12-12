"""Transaction extraction and categorization using LLM."""

import json
from typing import List, Dict, Any
from datetime import datetime
from llm_client import LLMClient
from config import AppConfig


class TransactionAnalyzer:
    """Extracts and categorizes financial transactions from text."""

    def __init__(self, llm_client: LLMClient):
        """Initialize with an LLM client."""
        self.llm_client = llm_client

    def extract_transactions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract transactions from document text using LLM.

        Args:
            text: Document text containing financial information

        Returns:
            List of transaction dictionaries
        """
        prompt = self._create_extraction_prompt(text)

        messages = [
            {
                "role": "system",
                "content": "You are a financial document analyzer. Extract transaction data accurately and return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )

            response_text = self.llm_client.get_response_text(response)

            # Parse JSON response
            transactions = self._parse_transactions(response_text)

            # Auto-categorize transactions
            for transaction in transactions:
                if not transaction.get("category"):
                    transaction["category"] = self.categorize_transaction(
                        transaction.get("description", "")
                    )

            return transactions

        except Exception as e:
            print(f"Error extracting transactions: {e}")
            return []

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for transaction extraction."""
        return f"""Analyze the following financial document and extract ALL transactions.

For each transaction, extract:
- date: Transaction date (format: YYYY-MM-DD, estimate if unclear)
- description: What the transaction was for
- amount: The amount (positive for income, negative for expenses)
- type: "income" or "expense"
- category: Best matching category from the lists below

Income Categories: {', '.join(AppConfig.INCOME_CATEGORIES)}
Expense Categories: {', '.join(AppConfig.EXPENSE_CATEGORIES)}

Document Text:
{text[:3000]}

Return ONLY a valid JSON array of transactions. Example format:
[
  {{
    "date": "2024-01-15",
    "description": "Grocery Store",
    "amount": -85.50,
    "type": "expense",
    "category": "Food & Dining"
  }},
  {{
    "date": "2024-01-01",
    "description": "Salary Deposit",
    "amount": 3500.00,
    "type": "income",
    "category": "Salary"
  }}
]

If no clear transactions found, return an empty array: []
"""

    def _parse_transactions(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse transactions from LLM response."""
        try:
            # Try to find JSON array in response
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                transactions = json.loads(json_str)

                # Validate and clean transactions
                valid_transactions = []
                for txn in transactions:
                    if self._validate_transaction(txn):
                        valid_transactions.append(txn)

                return valid_transactions

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")

        return []

    def _validate_transaction(self, txn: Dict[str, Any]) -> bool:
        """Validate transaction data."""
        required_fields = ["date", "description", "amount", "type"]
        return all(field in txn for field in required_fields)

    def categorize_transaction(self, description: str) -> str:
        """
        Categorize a single transaction based on description.

        Args:
            description: Transaction description

        Returns:
            Category name
        """
        description_lower = description.lower()

        # Simple keyword-based categorization
        category_keywords = {
            "Housing": ["rent", "mortgage", "property", "hoa"],
            "Transportation": ["gas", "fuel", "uber", "lyft", "taxi", "parking", "car", "auto"],
            "Food & Dining": ["restaurant", "food", "grocery", "cafe", "starbucks", "dining"],
            "Utilities": ["electric", "water", "internet", "phone", "gas bill", "utility"],
            "Healthcare": ["doctor", "hospital", "pharmacy", "medical", "health"],
            "Entertainment": ["movie", "netflix", "spotify", "game", "entertainment"],
            "Shopping": ["amazon", "store", "shop", "retail", "purchase"],
            "Subscriptions": ["subscription", "membership", "monthly"],
            "Travel": ["hotel", "flight", "airbnb", "travel"],
            "Salary": ["salary", "paycheck", "wage"],
            "Investment Income": ["dividend", "interest", "investment"],
        }

        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return category

        return "Other"

    def summarize_transactions(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Summary statistics
        """
        if not transactions:
            return {
                "total_income": 0,
                "total_expenses": 0,
                "net": 0,
                "transaction_count": 0,
                "by_category": {}
            }

        total_income = sum(
            txn["amount"] for txn in transactions
            if txn["type"] == "income"
        )

        total_expenses = sum(
            abs(txn["amount"]) for txn in transactions
            if txn["type"] == "expense"
        )

        # Group by category
        by_category = {}
        for txn in transactions:
            category = txn.get("category", "Other")
            if category not in by_category:
                by_category[category] = {
                    "count": 0,
                    "total": 0,
                    "transactions": []
                }

            by_category[category]["count"] += 1
            by_category[category]["total"] += abs(txn["amount"])
            by_category[category]["transactions"].append(txn)

        return {
            "total_income": round(total_income, 2),
            "total_expenses": round(total_expenses, 2),
            "net": round(total_income - total_expenses, 2),
            "transaction_count": len(transactions),
            "by_category": by_category
        }

    def generate_insights(
        self,
        transactions: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> str:
        """
        Generate AI insights about spending patterns.

        Args:
            transactions: List of transactions
            summary: Summary statistics

        Returns:
            AI-generated insights text
        """
        prompt = f"""Analyze these financial transactions and provide insights:

Summary:
- Total Income: ${summary['total_income']:.2f}
- Total Expenses: ${summary['total_expenses']:.2f}
- Net: ${summary['net']:.2f}
- Number of Transactions: {summary['transaction_count']}

Spending by Category:
{self._format_categories(summary['by_category'])}

Recent Transactions (last 5):
{self._format_transactions(transactions[-5:])}

Provide:
1. Key spending patterns and trends
2. Top spending categories
3. Potential areas for savings
4. Budget recommendations

Keep it concise and actionable (2-3 paragraphs)."""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful financial advisor providing insights on spending patterns."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return self.llm_client.get_response_text(response)

        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def _format_categories(self, by_category: Dict[str, Any]) -> str:
        """Format category breakdown for prompt."""
        lines = []
        for category, data in sorted(
            by_category.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        ):
            lines.append(f"- {category}: ${data['total']:.2f} ({data['count']} transactions)")
        return "\n".join(lines[:10])  # Top 10 categories

    def _format_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """Format transactions for prompt."""
        lines = []
        for txn in transactions:
            lines.append(
                f"- {txn['date']}: {txn['description']} - ${abs(txn['amount']):.2f} ({txn['category']})"
            )
        return "\n".join(lines)
