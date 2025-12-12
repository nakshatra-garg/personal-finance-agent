"""Personal Finance Agent - Streamlit Application."""

import streamlit as st
import os
from pathlib import Path
import json

from config import LLMConfig, AppConfig
from llm_client import get_llm_client
from document_processor import DocumentProcessor
from transaction_analyzer import TransactionAnalyzer
from budget_manager import BudgetManager
from rag_system import RAGSystem


# Page config
st.set_page_config(
    page_title="Personal Finance Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.transactions = []
    st.session_state.documents = []
    st.session_state.budget_manager = BudgetManager()
    st.session_state.rag_system = None
    st.session_state.chat_history = []


def init_llm_client():
    """Initialize LLM client based on sidebar selection."""
    provider = st.session_state.get("selected_provider")
    model = st.session_state.get("selected_model")

    if provider and model:
        try:
            return get_llm_client(provider, model)
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return None
    return None


def sidebar():
    """Render sidebar with configuration."""
    st.sidebar.title("⚙️ Configuration")

    # LLM Provider Selection
    st.sidebar.subheader("LLM Provider")

    available_providers = LLMConfig.get_available_providers()

    if not available_providers:
        st.sidebar.error("No API keys configured! Please set up your .env file.")
        st.sidebar.info("Copy .env.example to .env and add your API keys.")
        return False

    # Provider selection
    provider_names = {p: LLMConfig.PROVIDERS[p]["name"] for p in available_providers}
    selected_provider = st.sidebar.selectbox(
        "Provider",
        options=available_providers,
        format_func=lambda x: provider_names[x],
        key="selected_provider"
    )

    # Model selection
    models = LLMConfig.get_models(selected_provider)
    selected_model = st.sidebar.selectbox(
        "Model",
        options=models,
        key="selected_model"
    )

    st.sidebar.divider()

    # App Features
    st.sidebar.subheader("Features")
    st.sidebar.info(
        """
        📊 **Budget Tracking**
        📄 **Document Analysis**
        💬 **AI Chat Assistant**
        🔍 **Transaction Extraction**
        """
    )

    st.sidebar.divider()

    # Clear data button
    if st.sidebar.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.transactions = []
        st.session_state.documents = []
        st.session_state.budget_manager = BudgetManager()
        st.session_state.rag_system = None
        st.session_state.chat_history = []
        st.rerun()

    return True


def upload_documents_tab():
    """Document upload and processing tab."""
    st.header("📄 Upload Financial Documents")

    st.info("Upload bank statements, receipts, invoices, or any financial documents (PDF, DOCX, TXT, CSV, XLSX)")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=AppConfig.SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            process_documents(uploaded_files)


def process_documents(uploaded_files):
    """Process uploaded documents."""
    llm_client = init_llm_client()

    if not llm_client:
        st.error("Please configure LLM provider first.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize RAG system if not exists
    if st.session_state.rag_system is None:
        st.session_state.rag_system = RAGSystem(llm_client)

    transaction_analyzer = TransactionAnalyzer(llm_client)
    doc_processor = DocumentProcessor()

    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}...")

        # Save file temporarily
        temp_path = Path(f"temp_{uploaded_file.name}")
        temp_path.write_bytes(uploaded_file.read())

        try:
            # Extract text
            result = doc_processor.process_file(str(temp_path))

            if result["success"]:
                # Add to RAG system
                st.session_state.rag_system.add_document(
                    result["text"],
                    result["metadata"]
                )

                # Extract transactions
                transactions = transaction_analyzer.extract_transactions(result["text"])

                if transactions:
                    st.session_state.transactions.extend(transactions)
                    st.session_state.budget_manager.add_transactions(transactions)

                st.session_state.documents.append({
                    "name": uploaded_file.name,
                    "metadata": result["metadata"],
                    "transaction_count": len(transactions)
                })

            else:
                st.error(f"Failed to process {uploaded_file.name}: {result['error']}")

        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    progress_bar.empty()
    status_text.empty()

    st.success(f"✅ Processed {total_files} documents successfully!")
    st.rerun()


def transactions_tab():
    """Display and manage transactions."""
    st.header("💳 Transactions")

    if not st.session_state.transactions:
        st.info("No transactions yet. Upload documents to extract transactions.")
        return

    # Summary metrics
    analyzer = TransactionAnalyzer(init_llm_client())
    summary = analyzer.summarize_transactions(st.session_state.transactions)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Income", f"${summary['total_income']:,.2f}")
    with col2:
        st.metric("Total Expenses", f"${summary['total_expenses']:,.2f}")
    with col3:
        st.metric("Net", f"${summary['net']:,.2f}")
    with col4:
        st.metric("Transactions", summary['transaction_count'])

    st.divider()

    # Display by category
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Spending by Category")
        if summary['by_category']:
            for category, data in sorted(
                summary['by_category'].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:10]:
                st.write(f"**{category}**: ${data['total']:,.2f} ({data['count']} transactions)")

    with col2:
        st.subheader("Recent Transactions")
        for txn in reversed(st.session_state.transactions[-10:]):
            amount_color = "green" if txn['type'] == 'income' else "red"
            st.write(
                f"{txn['date']} - {txn['description']} - "
                f":{amount_color}[${abs(txn['amount']):,.2f}] ({txn['category']})"
            )

    st.divider()

    # AI Insights
    if st.button("Generate AI Insights", type="primary"):
        with st.spinner("Analyzing spending patterns..."):
            insights = analyzer.generate_insights(
                st.session_state.transactions,
                summary
            )
            st.subheader("💡 AI Insights")
            st.write(insights)


def budget_tab():
    """Budget management tab."""
    st.header("📊 Budget Management")

    budget_mgr = st.session_state.budget_manager

    # Budget setup
    st.subheader("Set Up Budget")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Quick Setup: Enter your monthly income**")
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            key="monthly_income"
        )

        savings_rate = st.slider(
            "Savings Rate (%)",
            min_value=0,
            max_value=50,
            value=20,
            key="savings_rate"
        ) / 100

        if st.button("Generate Budget Suggestion", type="primary"):
            suggested = budget_mgr.suggest_budget(monthly_income, savings_rate)
            st.session_state.suggested_budget = suggested
            st.success("Budget suggestion generated!")

    with col2:
        st.info(
            """
            **Budget Rules:**

            50% - Needs
            30% - Wants
            20% - Savings
            """
        )

    # Apply suggested budget
    if "suggested_budget" in st.session_state:
        st.divider()
        st.subheader("Suggested Budget")

        if st.button("Apply This Budget"):
            for category, amount in st.session_state.suggested_budget.items():
                budget_mgr.set_budget(category, amount)
            st.success("Budget applied!")
            st.rerun()

        # Display suggestion
        cols = st.columns(3)
        for idx, (category, amount) in enumerate(st.session_state.suggested_budget.items()):
            with cols[idx % 3]:
                st.metric(category, f"${amount:,.2f}")

    # Current budget status
    if budget_mgr.budgets:
        st.divider()
        st.subheader("Current Budget Status")

        summary = budget_mgr.get_budget_summary()

        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Budget", f"${summary['total_allocated']:,.2f}")
        with col2:
            st.metric("Total Spent", f"${summary['total_spent']:,.2f}")
        with col3:
            st.metric("Remaining", f"${summary['total_remaining']:,.2f}")

        # Category status
        st.subheader("Category Breakdown")

        for cat in summary['categories']:
            status_emoji = {
                "good": "✅",
                "warning": "⚠️",
                "over": "🚨"
            }.get(cat['status'], "")

            with st.expander(f"{status_emoji} {cat['name']} - {cat['percent_used']:.1f}% used"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Allocated", f"${cat['allocated']:,.2f}")
                with col2:
                    st.metric("Spent", f"${cat['spent']:,.2f}")
                with col3:
                    st.metric("Remaining", f"${cat['remaining']:,.2f}")

                st.progress(min(cat['percent_used'] / 100, 1.0))

        # Alerts
        alerts = budget_mgr.get_overspending_alerts()
        if alerts:
            st.divider()
            st.subheader("⚠️ Alerts")
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(
                        f"🚨 **{alert['category']}**: Over budget by ${alert['over_by']:,.2f}!"
                    )
                else:
                    st.warning(
                        f"⚠️ **{alert['category']}**: {alert['percent_used']:.1f}% of budget used"
                    )


def chat_tab():
    """AI chat assistant tab."""
    st.header("💬 AI Financial Assistant")

    llm_client = init_llm_client()

    if not llm_client:
        st.error("Please configure LLM provider first.")
        return

    # Initialize RAG if documents exist
    if st.session_state.rag_system and st.session_state.documents:
        st.info(f"📚 {len(st.session_state.documents)} documents loaded. Ask questions about your finances!")

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your finances..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use RAG if available and relevant
                if st.session_state.rag_system and st.session_state.documents:
                    response = st.session_state.rag_system.query(prompt)
                else:
                    # General financial advice
                    messages = [
                        {"role": "system", "content": "You are a helpful personal finance advisor."},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-5:]]
                    ]
                    llm_response = llm_client.chat_completion(messages, temperature=0.7)
                    response = llm_client.get_response_text(llm_response)

                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


def main():
    """Main application."""
    st.title("💰 Personal Finance Agent")

    # Sidebar
    if not sidebar():
        st.error("Please configure your API keys in .env file to use this app.")
        st.code("""
# Copy .env.example to .env and add your keys:
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
# etc.
        """)
        return

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Upload Documents",
        "💳 Transactions",
        "📊 Budget",
        "💬 AI Chat"
    ])

    with tab1:
        upload_documents_tab()

    with tab2:
        transactions_tab()

    with tab3:
        budget_tab()

    with tab4:
        chat_tab()


if __name__ == "__main__":
    main()
