<p align="center">
  <img src="logo.png" alt="SantAI Logo" width="400"/>
</p>

# SantAI - AI Financial Assistant for Debt Management

[![Meta Llama](https://img.shields.io/badge/Meta-Llama-FF6B35?style=for-the-badge&logo=meta&logoColor=white)](https://llama.meta.com/)
[![Groq](https://img.shields.io/badge/Groq-00A67E?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

SantAI is an empathetic AI-powered financial assistant designed to help Indonesian borrowers navigate debt challenges safely and calmly. By leveraging official OJK (Otoritas Jasa Keuangan) regulations and real-time AI analysis, SantAI provides immediate clarity on loan legality and personalized guidance for debt management.

## 🚀 Key Features

- **🔍 Legal Status Verification**: Instantly checks if lenders are registered with OJK or operate illegally
- **💬 Empathetic Counseling**: Provides calm, non-judgmental guidance to reduce financial stress
- **📊 Financial Health Analysis**: Calculates debt-to-income ratios and assesses repayment capacity
- **🛡️ Safety-First Approach**: Cites official regulatory documents for trustworthy advice
- **🔒 Privacy-Focused**: All conversations are confidential and user data is handled securely
- **🌐 Multi-Platform Ready**: Built on Streamlit for easy web deployment with WhatsApp integration potential

## 🏗️ How It Works

SantAI combines regulatory intelligence with AI-powered analysis:

1. **User Onboarding**: Collects user profile and loan details through conversational interface
2. **Regulatory Matching**: Cross-references lender information with OJK's official database
3. **Legal Status Determination**: Identifies legitimate vs. illegal lending operations
4. **Personalized Analysis**: Provides tailored advice based on financial situation and loan legality
5. **Actionable Guidance**: Offers clear next steps for both legal and illegal loan scenarios

## 📦 Installation

### Prerequisites

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))
- OJK regulatory data file (Updated regularly)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/santai.git
cd santai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Add your Groq API key to .env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Prepare regulatory data**
```bash
mkdir data
# Place your OJK regulatory file as data/peraturan.txt
```

5. **Run the application**
```bash
streamlit run santai.py
```

## 🛠️ Technical Architecture

### Core Components

- **AI Engine**: Groq-powered LLaMA 3.1 8B Instant for fast, intelligent responses
- **Knowledge Base**: FAISS vector store with OJK regulatory documents
- **Embeddings**: HuggingFace Sentence Transformers for semantic search
- **Web Interface**: Streamlit for responsive, user-friendly experience
- **Data Processing**: Custom text splitting and similarity matching algorithms

### Key Dependencies

```python
streamlit>=1.28.0
langchain-groq>=0.1.0
langchain>=0.1.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
python-Levenshtein>=0.25.0
python-dotenv>=1.0.0
```

## 📁 Project Structure

```
santai/
├── santai.py                # Main application file
├── data/
│   └── peraturan.txt        # OJK regulatory documents
├── logo.png                 # Application logo
├── requirements.txt         # Python dependencies
├── .env                     # Environment for Groq API key (gitignored)
└── README.md                # This file
```

## 🎯 Business Model

### Value Proposition

**For Borrowers:**
- Immediate clarity on loan legality and risks
- Empathetic guidance to reduce stress and prevent harmful decisions
- Actionable steps for dealing with both legal and illegal lenders
- Connections to trusted financial institutions for safe solutions

**For Regulators:**
- Citizen-facing tool that strengthens enforcement against illegal lenders
- Improved reporting and data collection on illegal lending activities
- Public education platform for financial literacy

### Revenue Streams

- **Referral Partnerships**: Fees from licensed financial institutions for debt consolidation referrals
- **CSR Funding**: Sponsorships from NGOs and development agencies
- **Premium Services**: Advanced financial consultation and legal support
- **Educational Partnerships**: Financial literacy programs and workshops

## 🌟 Why SantAI?

### Problem Being Solved

Indonesia faces a significant challenge with illegal online lenders (pinjol ilegal) that use aggressive collection tactics and charge predatory interest rates. Many borrowers lack access to reliable information about lender legitimacy and proper debt management guidance.

### Our Solution

SantAI bridges this gap by providing:
- **Instant Verification**: Real-time checking against OJK's official records
- **Psychological Safety**: Empathetic tone reduces shame and stress
- **Regulatory Compliance**: Always cites official sources and procedures
- **Accessibility**: Free, available 24/7 without judgment

## 🤝 Partnerships & Collaboration

We welcome partnerships with:
- **Regulatory Bodies**: OJK, Satgas PASTI for data validation
- **Financial Institutions**: Licensed lenders for safe refinancing options
- **NGOs & Consumer Protection Groups**: Awareness campaigns and outreach
- **Academic Institutions**: Research collaboration and validation studies

## 🔒 Privacy & Security

- **Data Minimization**: Only collect essential information needed for analysis
- **Local Processing**: Sensitive data processed locally when possible
- **Transparent Operations**: Clear privacy policy and data handling practices
- **Regulatory Compliance**: Adherence to Indonesian financial data protection laws

## 🚀 Future Roadmap

- [ ] WhatsApp chatbot integration for broader accessibility
- [ ] Mobile app development (iOS/Android)
- [ ] Advanced financial planning tools
- [ ] Multi-language support (English, local languages)
- [ ] Integration with financial management APIs
- [ ] Real-time regulatory updates directly from OJK

## 👥 Contributing

We welcome contributions from developers, financial experts, and community advocates. Please contact us for more details.

### Areas for Contribution
- Regulatory data collection and validation
- UI/UX improvements for better user experience
- Additional language support
- Integration with financial APIs
- Documentation and translation

## 📄 License

This project is built using Meta's Llama models and is subject to the [Llama Community License](https://llama.meta.com/llama3/license/). 

**Key restrictions include:**
- Commercial use limited to services with fewer than 700 million monthly active users
- Required attribution to Meta
- Prohibition on certain illegal or harmful uses
- Compliance with applicable laws and regulations

Please review the full license terms before using, modifying, or distributing this software.

## 📞 Support & Contact

For technical support, partnership inquiries, or regulatory collaboration:
- **Email**: ladityarsa.ian@gmail.com ; bilsap99@gmail.com
- **Issues**: [GitHub Issues](https://github.com/IanKuzuma/SantAI/issues)
- **Documentation**: [Project Wiki](https://github.com/IanKuzuma/SantAI/wiki)

## 🙏 Acknowledgments

- **OJK (Otoritas Jasa Keuangan)** for regulatory guidance and oversight
- **Meta** for the Llama language models that power our AI capabilities
- **Groq** for high-performance AI inference capabilities
- **Hugging Face** for open-source AI models and embeddings
- **Streamlit** for rapid web application development
- **Satgas PASTI** for their work combating illegal online lending

---

<div align="center">
  
*SantAI - Making financial guidance accessible, empathetic, and safe for every Indonesian borrower*

**"Tenang, Aman, dan Tidak Panik"** - Calm, Safe, and Don't Panic

</div>
