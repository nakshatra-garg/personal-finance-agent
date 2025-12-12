# Personal Finance Agent

A comprehensive AI-powered personal finance management tool built with Streamlit. Analyze financial documents, track expenses, manage budgets, and get intelligent insights about your spending patterns.

## Features

### Multi-LLM Support
- **Groq** (Primary)
- **OpenAI** (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- **Azure OpenAI**
- **Perplexity**

Easily switch between providers and models through the UI.

### Document Processing
Upload and analyze financial documents:
- **PDF** - Bank statements, invoices
- **DOCX/DOC** - Financial reports, receipts
- **TXT** - Plain text financial data
- **CSV** - Transaction exports
- **XLSX/XLS** - Spreadsheets

### Smart Transaction Extraction
- Automatically extracts transactions from uploaded documents
- AI-powered categorization (Housing, Food, Transportation, etc.)
- Date and amount parsing
- Income vs. expense classification

### Budget Management
- Set budgets by category
- Track spending in real-time
- Visual progress indicators
- Overspending alerts
- Smart budget suggestions based on income (50/30/20 rule)

### RAG-Powered Q&A
- Upload documents and ask questions about them
- Semantic search across all uploaded documents
- Context-aware responses
- Source citations

### AI Chat Assistant
- Get financial advice
- Ask questions about your spending
- Receive personalized insights
- Budget recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- API key for at least one LLM provider (Groq recommended)

### Setup

1. Clone or download this repository:
```bash
cd fin_agent
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use any text editor
```

Example `.env` configuration:
```env
# At minimum, configure one provider:
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional: Add other providers
OPENAI_API_KEY=sk-your_openai_key_here
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
PERPLEXITY_API_KEY=your_perplexity_key_here
```

### Getting API Keys

#### Groq (Recommended - Fast & Free Tier)
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Go to API Keys section
4. Create a new API key

#### OpenAI
1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up and add payment method
3. Go to API Keys
4. Create new secret key

#### Azure OpenAI
1. Access through Azure Portal
2. Create Azure OpenAI resource
3. Get endpoint and API key from resource

#### Perplexity
1. Visit [perplexity.ai](https://www.perplexity.ai)
2. Sign up for API access
3. Generate API key

## Usage

### Start the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Workflow

1. **Configure LLM Provider**
   - Use the sidebar to select your LLM provider and model
   - The app will only show providers with valid API keys

2. **Upload Documents**
   - Go to "Upload Documents" tab
   - Upload financial documents (statements, receipts, invoices)
   - Click "Process Documents"
   - The AI will extract transactions and add to RAG system

3. **View Transactions**
   - Navigate to "Transactions" tab
   - See all extracted transactions
   - View spending by category
   - Generate AI insights about your spending patterns

4. **Manage Budget**
   - Go to "Budget" tab
   - Enter your monthly income
   - Generate budget suggestions (50/30/20 rule)
   - Apply suggested budget or customize
   - Track spending vs. budget in real-time
   - Get alerts for overspending

5. **Chat with AI**
   - Navigate to "AI Chat" tab
   - Ask questions about your finances
   - Get insights from uploaded documents
   - Receive personalized financial advice

## Project Structure

```
fin_agent/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and constants
├── llm_client.py              # Unified LLM client interface
├── document_processor.py      # Document text extraction
├── transaction_analyzer.py    # Transaction extraction & analysis
├── budget_manager.py          # Budget tracking logic
├── rag_system.py              # RAG implementation for Q&A
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment configuration
├── .gitignore                # Git ignore rules
└── README.md                  # This file
```

## Features in Detail

### Transaction Extraction
The AI analyzes your documents and extracts:
- Transaction dates
- Descriptions
- Amounts (positive for income, negative for expenses)
- Automatic categorization
- Transaction type (income/expense)

### Budget Categories
Default categories include:
- **Needs**: Housing, Transportation, Food & Dining, Utilities, Healthcare
- **Wants**: Entertainment, Shopping, Personal Care, Subscriptions
- **Savings**: Savings, Investments
- **Other**: Education, Insurance, Debt Payments, Travel

### RAG System
- Uses `sentence-transformers` for embeddings
- Chunks documents for efficient retrieval
- Semantic search finds relevant information
- Provides source citations

## Troubleshooting

### "No API keys configured"
- Make sure you copied `.env.example` to `.env`
- Add at least one valid API key
- Restart the application

### Document processing fails
- Check file size (limit: 10MB)
- Ensure file format is supported
- Try simpler documents first

### Transactions not extracted
- Ensure document contains clear transaction data
- Try adjusting the LLM model (GPT-4 works better for complex docs)
- Check if amounts and dates are in standard formats

### RAG not working
- Make sure documents are uploaded first
- Wait for document processing to complete
- Try more specific questions

## Privacy & Security

- **Session-only storage**: All data is cleared when you close the browser
- **No cloud storage**: Documents are processed locally in memory
- **API keys**: Stored locally in `.env` file (never commit this file)
- **No data retention**: No transaction or document data is persisted

## Limitations

- Session-only storage (no persistence)
- Single-user application
- Document size limit: 10MB per file
- Requires internet connection for LLM API calls

## Future Enhancements

Potential improvements:
- Database persistence (SQLite)
- Multi-user support
- Data export (CSV, JSON)
- Charts and visualizations
- Recurring transaction detection
- Bill payment reminders
- Investment portfolio tracking
- Tax planning features

## Contributing

This is a personal project. Feel free to fork and customize for your needs!

## License

MIT License - Feel free to use and modify as needed.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- LLM providers: Groq, OpenAI, Azure, Perplexity
- Document processing: PyPDF2, python-docx, pandas
- Embeddings: sentence-transformers

## Support

For issues or questions:
1. Check this README
2. Review error messages in the app
3. Verify API keys are configured correctly
4. Check console output for detailed errors

---

Built with Claude Code
