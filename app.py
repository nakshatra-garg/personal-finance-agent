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
    st.session_state.api_keys = {}
    st.session_state.api_endpoints = {}
    st.session_state.selected_provider = None
    st.session_state.selected_model = None


def test_api_key(provider: str, api_key: str, endpoint: str = None) -> bool:
    """Test if an API key is valid by making a simple call."""
    try:
        session_keys = {provider: api_key}
        session_endpoints = {f"{provider}_endpoint": endpoint} if endpoint else {}

        client = get_llm_client(
            provider=provider,
            model=LLMConfig.get_default_model(provider),
            session_keys=session_keys,
            session_endpoints=session_endpoints
        )

        # Make a simple test call
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )

        return True
    except Exception as e:
        st.error(f"API key test failed: {str(e)}")
        return False


def configure_api_keys_ui():
    """Render API key configuration UI."""
    st.sidebar.subheader("🔑 API Configuration")

    st.sidebar.warning("⚠️ **Session Only:** Keys entered here are NOT saved permanently. They'll be cleared when you close the browser.")
    st.sidebar.info("💡 **For permanent storage:** Use .env file instead")

    # Get existing keys from environment as defaults
    for provider, config in LLMConfig.PROVIDERS.items():
        with st.sidebar.expander(f"⚙️ {config['name']}", expanded=False):
            # Get existing key from session or env
            existing_key = st.session_state.api_keys.get(provider, "")
            if not existing_key:
                existing_key = os.getenv(config["api_key_env"], "")

            # API Key input
            api_key = st.text_input(
                "API Key",
                value=existing_key,
                type="password",
                key=f"{provider}_key_input",
                help=f"Enter your {config['name']} API key"
            )

            # Azure needs endpoint too
            endpoint = None
            if provider == "azure":
                existing_endpoint = st.session_state.api_endpoints.get(f"{provider}_endpoint", "")
                if not existing_endpoint:
                    existing_endpoint = os.getenv(config.get("endpoint_env", ""), "")

                endpoint = st.text_input(
                    "Endpoint URL",
                    value=existing_endpoint,
                    key=f"{provider}_endpoint_input",
                    help="Your Azure OpenAI endpoint URL"
                )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save", key=f"{provider}_save", use_container_width=True):
                    if api_key:
                        st.session_state.api_keys[provider] = api_key
                        if endpoint:
                            st.session_state.api_endpoints[f"{provider}_endpoint"] = endpoint
                        st.success("✅ Saved!")
                        st.rerun()
                    else:
                        st.warning("Please enter an API key")

            with col2:
                if st.button("🧪 Test", key=f"{provider}_test", use_container_width=True):
                    if api_key:
                        with st.spinner("Testing..."):
                            if test_api_key(provider, api_key, endpoint):
                                st.success("✅ Valid!")
                                # Auto-save if test succeeds
                                st.session_state.api_keys[provider] = api_key
                                if endpoint:
                                    st.session_state.api_endpoints[f"{provider}_endpoint"] = endpoint
                    else:
                        st.warning("Please enter an API key")

            # Show if configured
            if provider in st.session_state.api_keys and st.session_state.api_keys[provider]:
                st.success("✅ Configured")


def init_llm_client():
    """Initialize LLM client based on sidebar selection."""
    provider = st.session_state.get("selected_provider")
    model = st.session_state.get("selected_model")

    if provider and model:
        try:
            return get_llm_client(
                provider=provider,
                model=model,
                session_keys=st.session_state.api_keys,
                session_endpoints=st.session_state.api_endpoints
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return None
    return None


def sidebar():
    """Render sidebar with configuration."""
    st.sidebar.title("⚙️ Settings")

    # API Key Configuration Section
    configure_api_keys_ui()

    st.sidebar.divider()

    # LLM Provider Selection
    st.sidebar.subheader("🤖 Active LLM")

    available_providers = LLMConfig.get_available_providers(st.session_state.api_keys)

    if not available_providers:
        st.sidebar.error("⚠️ No API keys configured!")
        st.sidebar.info("👆 Configure at least one provider above")
        return False

    # Provider selection
    provider_names = {p: LLMConfig.PROVIDERS[p]["name"] for p in available_providers}

    # Set default if not set
    if not st.session_state.selected_provider or st.session_state.selected_provider not in available_providers:
        st.session_state.selected_provider = available_providers[0]

    selected_provider = st.sidebar.selectbox(
        "Provider",
        options=available_providers,
        format_func=lambda x: provider_names[x],
        key="provider_select",
        index=available_providers.index(st.session_state.selected_provider) if st.session_state.selected_provider in available_providers else 0
    )

    st.session_state.selected_provider = selected_provider

    # Model selection
    models = LLMConfig.get_models(selected_provider)

    # Set default model if not set
    if not st.session_state.selected_model or st.session_state.selected_model not in models:
        st.session_state.selected_model = LLMConfig.get_default_model(selected_provider)

    selected_model = st.sidebar.selectbox(
        "Model",
        options=models,
        key="model_select",
        index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
    )

    st.session_state.selected_model = selected_model

    st.sidebar.success(f"✅ Using: {provider_names[selected_provider]}")

    st.sidebar.divider()

    # App Features
    st.sidebar.subheader("✨ Features")
    st.sidebar.info(
        """
        📊 Budget Tracking
        📄 Document Analysis
        💬 AI Chat Assistant
        🔍 Transaction Extraction
        📈 Financial Insights
        """
    )

    st.sidebar.divider()

    # Session Stats
    if st.session_state.documents or st.session_state.transactions:
        st.sidebar.subheader("📊 Session Stats")
        st.sidebar.metric("Documents", len(st.session_state.documents))
        st.sidebar.metric("Transactions", len(st.session_state.transactions))

    # Clear data button
    if st.sidebar.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.transactions = []
        st.session_state.documents = []
        st.session_state.budget_manager = BudgetManager()
        st.session_state.rag_system = None
        st.session_state.chat_history = []
        st.success("✅ All data cleared!")
        st.rerun()

    return True


def upload_documents_tab():
    """Document upload and processing tab."""
    st.header("📄 Upload Financial Documents")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("📤 Upload bank statements, receipts, invoices, or any financial documents")

    with col2:
        st.metric("Processed Docs", len(st.session_state.documents))

    st.write("**Supported formats:** PDF, DOCX, TXT, CSV, XLSX")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=AppConfig.SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected**")
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
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
        status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{total_files})")

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
                status_text.text(f"Extracting transactions from {uploaded_file.name}...")
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

    st.success(f"✅ Successfully processed {total_files} document(s)!")

    # Show summary
    total_transactions = sum(doc['transaction_count'] for doc in st.session_state.documents)
    if total_transactions > 0:
        st.info(f"📊 Extracted {total_transactions} transactions total")

    st.rerun()


def transactions_tab():
    """Display and manage transactions."""
    st.header("💳 Transactions")

    if not st.session_state.transactions:
        st.info("📭 No transactions yet. Upload documents to extract transactions.")
        return

    # Summary metrics
    analyzer = TransactionAnalyzer(init_llm_client())
    summary = analyzer.summarize_transactions(st.session_state.transactions)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Income", f"${summary['total_income']:,.2f}", delta=None)
    with col2:
        st.metric("💸 Total Expenses", f"${summary['total_expenses']:,.2f}", delta=None)
    with col3:
        net_value = summary['net']
        st.metric("📊 Net", f"${net_value:,.2f}", delta=f"${net_value:,.2f}")
    with col4:
        st.metric("🔢 Count", summary['transaction_count'])

    st.divider()

    # Display by category
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Spending by Category")
        if summary['by_category']:
            category_data = sorted(
                summary['by_category'].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:10]

            for category, data in category_data:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{category}**")
                with col_b:
                    st.write(f"${data['total']:,.2f}")
                st.progress(min(data['total'] / summary['total_expenses'], 1.0) if summary['total_expenses'] > 0 else 0)

    with col2:
        st.subheader("📝 Recent Transactions")
        for txn in reversed(st.session_state.transactions[-10:]):
            amount_color = "🟢" if txn['type'] == 'income' else "🔴"
            st.write(
                f"{amount_color} **{txn['date']}** - {txn['description']}"
            )
            st.caption(f"${abs(txn['amount']):,.2f} • {txn['category']}")
            st.divider()

    st.divider()

    # AI Insights
    if st.button("💡 Generate AI Insights", type="primary"):
        with st.spinner("🤔 Analyzing spending patterns..."):
            insights = analyzer.generate_insights(
                st.session_state.transactions,
                summary
            )
            st.subheader("💡 AI Financial Insights")
            st.write(insights)


def budget_tab():
    """Budget management tab."""
    st.header("📊 Budget Management")

    budget_mgr = st.session_state.budget_manager

    # Budget setup
    st.subheader("💰 Set Up Budget")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Quick Setup: 50/30/20 Rule**")
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

        if st.button("✨ Generate Budget Suggestion", type="primary"):
            suggested = budget_mgr.suggest_budget(monthly_income, savings_rate)
            st.session_state.suggested_budget = suggested
            st.success("✅ Budget suggestion generated!")
            st.rerun()

    with col2:
        st.info(
            """
            **50/30/20 Rule:**

            • 50% - Needs
            • 30% - Wants
            • 20% - Savings

            Adjust as needed!
            """
        )

    # Apply suggested budget
    if "suggested_budget" in st.session_state:
        st.divider()
        st.subheader("💡 Suggested Budget Allocation")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Based on **${monthly_income:,.2f}** monthly income with **{int(savings_rate*100)}%** savings rate")
        with col2:
            if st.button("✅ Apply This Budget", type="primary", use_container_width=True):
                for category, amount in st.session_state.suggested_budget.items():
                    budget_mgr.set_budget(category, amount)
                st.success("✅ Budget applied successfully!")
                st.rerun()

        # Display suggestion in grid
        cols = st.columns(4)
        for idx, (category, amount) in enumerate(st.session_state.suggested_budget.items()):
            with cols[idx % 4]:
                st.metric(category, f"${amount:,.2f}")

    # Current budget status
    if budget_mgr.budgets:
        st.divider()
        st.subheader("📈 Current Budget Status")

        summary = budget_mgr.get_budget_summary()

        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Total Budget", f"${summary['total_allocated']:,.2f}")
        with col2:
            st.metric("💸 Total Spent", f"${summary['total_spent']:,.2f}")
        with col3:
            remaining = summary['total_remaining']
            st.metric("💵 Remaining", f"${remaining:,.2f}")

        # Category status
        st.subheader("📊 Category Breakdown")

        for cat in summary['categories']:
            status_emoji = {
                "good": "✅",
                "warning": "⚠️",
                "over": "🚨"
            }.get(cat['status'], "")

            with st.expander(f"{status_emoji} {cat['name']} - {cat['percent_used']:.1f}% used", expanded=(cat['status'] != 'good')):
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
            st.subheader("⚠️ Budget Alerts")
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(
                        f"🚨 **{alert['category']}**: Over budget by **${alert['over_by']:,.2f}**!"
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
        st.error("⚠️ Please configure LLM provider in the sidebar first.")
        return

    # Initialize RAG if documents exist
    if st.session_state.rag_system and st.session_state.documents:
        st.info(f"📚 {len(st.session_state.documents)} document(s) loaded. Ask questions about your finances!")
    else:
        st.info("💡 Upload documents in the 'Upload Documents' tab to enable document Q&A")

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
            with st.spinner("🤔 Thinking..."):
                # Use RAG if available and relevant
                if st.session_state.rag_system and st.session_state.documents:
                    response = st.session_state.rag_system.query(prompt)
                else:
                    # General financial advice
                    messages = [
                        {"role": "system", "content": "You are a helpful personal finance advisor. Provide practical, actionable advice."},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history[-5:]]
                    ]
                    llm_response = llm_client.chat_completion(messages, temperature=0.7)
                    response = llm_client.get_response_text(llm_response)

                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


def main():
    """Main application."""
    st.title("💰 Personal Finance Agent")
    st.caption("AI-powered financial document analysis and budget management")

    # Sidebar
    if not sidebar():
        st.warning("⚠️ **Getting Started**")
        st.info(
            """
            👈 Configure at least one LLM provider in the sidebar to get started:

            1. Click on a provider (Groq, OpenAI, etc.)
            2. Enter your API key
            3. Click 'Test' to validate
            4. Start using the app!

            **Don't have an API key?** Get a free Groq API key at [console.groq.com](https://console.groq.com)
            """
        )
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
