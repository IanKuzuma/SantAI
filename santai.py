# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os
from dotenv import load_dotenv
import pypdf

# Load API Key
load_dotenv()
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = os.getenv('GROQ_API_KEY')

# Function to load and process our knowledge source
@st.cache_resource
def build_knowledge_base():
    documents = []
    
    # 1. Load data from text files with error handling
    try:
        with open('data/licensed_lenders.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Only add non-empty lines
                    documents.append(Document(page_content=f"Lender '{line}' is LICENSED and legal by OJK."))
    except FileNotFoundError:
        st.sidebar.error("licensed_lenders.txt not found. Please create this file.")
        return None

    try:
        with open('data/illegal_lenders.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(Document(page_content=f"Lender '{line}' is ILLEGAL and not licensed by OJK. Warning: Do not engage with this lender."))
    except FileNotFoundError:
        st.sidebar.error("illegal_lenders.txt not found. Please create this file.")
        return None

    # 2. Check if we have any documents
    if not documents:
        st.sidebar.error("No documents were loaded. Please check your data files.")
        return None

    # 3. Create Embeddings with better error handling
    try:
        # Use a more reliable embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Test the embedding model with a simple text
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) == 0:
            st.sidebar.error("Embedding model failed to generate vectors.")
            return None
            
    except Exception as e:
        st.sidebar.error(f"Error loading embedding model: {e}")
        return None

    # 4. Create Vector Store
    try:
        vectorstore = Chroma.from_documents(documents, embeddings)
        st.sidebar.success("Knowledge base loaded successfully!")
        return vectorstore
    except Exception as e:
        st.sidebar.error(f"Error creating vector store: {e}")
        return None


# Build the knowledge base
vectorstore = build_knowledge_base()


# Initialize the Groq LLM (Let's use the fast Llama 3 8B for prototyping)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")


# Initialize session state for user data and conversation flow
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.user_data = {
        'name': None, 'age': None, 'job': None, 'income': None,
        'lenders': [],  # List to store multiple lenders
        'data_complete': False
    }
    st.session_state.current_step = 'name'
    st.session_state.analysis_shown = False
    st.session_state.should_generate_analysis = False
    st.session_state.analyzing_msg = None
    st.session_state.current_lender = {}  # Temporary storage for current lender
    st.session_state.add_another_lender = None  # Track if user wants to add more


# Set page config
st.set_page_config(
    page_title="SantAI - AI yang bikin tenang, aman, dan nggak panik ketika terlilit utang!", 
    page_icon="logo.png",
    layout="wide")

# Create columns for logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=256)  # Adjust width as needed
with col2:
    st.header("SantAI - Teman anda untuk bikin tenang, aman, dan nggak panik ketika terlilit utang!")
    st.caption("Powered by LLaMA via Groq & OJK Data")

# Alternatively, in sidebar
with st.sidebar:
    st.image("logo.png", width=100)
    st.text("AI yang bikin tenang, aman, dan nggak panik ketika terlilit utang!")


def get_next_question():
    """Determine which question to ask next based on current step"""
    current_step = st.session_state.current_step
    
    # Define the flow for the first lender
    if current_step == 'name':
        return 'age'
    elif current_step == 'age':
        return 'job'
    elif current_step == 'job':
        return 'income'
    elif current_step == 'income':
        return 'lender_name'
    elif current_step == 'lender_name':
        return 'lender_website'
    elif current_step == 'lender_website':
        return 'loan_amount'
    elif current_step == 'loan_amount':
        return 'monthly_installment'
    elif current_step == 'monthly_installment':
        return 'add_another_lender'
    
    # Handle the "add another lender" decision
    elif current_step == 'add_another_lender':
        # This should be handled in the input processing, not here
        return 'add_another_lender'  # Stay here until user answers
    
    # Handle subsequent lenders
    elif current_step == 'next_lender_name':
        return 'lender_website'
    
    elif current_step == 'complete':
        return 'complete'
    
    # Default case
    return 'complete'

def ask_question(question_type):
    """Ask the appropriate question based on type"""
    questions = {
        'name': "Selamat datang! Saya SantAI, asisten finansial Anda. Sebelum mulai, boleh saya tahu nama Anda?",
        'age': f"Terima kasih, {st.session_state.user_data['name']}! Berapa usia Anda?",
        'job': f"Usia {st.session_state.user_data['age']} tahun. Apa pekerjaan Anda saat ini?",
        'income': f"Pekerjaan sebagai {st.session_state.user_data['job']}. Berapa penghasilan bulanan Anda? (dalam Rupiah)",
        'lender_name': "Terima kasih atas informasinya. Sekarang, mari kita bahas tentang pinjaman Anda. Siapa nama pemberi pinjaman?",
        'lender_website': "Apakah pinjaman tersebut memiliki website? (Jika ada, sebutkan. Jika tidak, ketik 'Tidak')",
        'loan_amount': "Berapa total jumlah pinjaman yang Anda ambil dari lender ini? (dalam Rupiah)",
        'monthly_installment': "Berapa besar angsuran bulanan yang harus Anda bayarkan untuk lender ini? (dalam Rupiah)",
        'add_another_lender': f"Apakah Anda memiliki pinjaman dari lender lain yang ingin ditambahkan? (Jawab dengan: Ya atau Tidak)",
        'next_lender_name': "Siapa nama pemberi pinjaman lainnya?",
        'complete': "Terima kasih! Data semua lender telah tercatat."
    }
    return questions.get(question_type, "Terima kasih! Data Anda sudah lengkap.")

def extract_number(text):
    """Extract numbers from text input, handling various formats"""
    try:
        # Remove all non-digit characters except commas and dots
        cleaned = ''.join(c for c in text if c.isdigit() or c in [',', '.'])
        # Remove commas and convert to float, then to int
        cleaned = cleaned.replace(',', '').replace('.', '')
        return int(cleaned)
    except:
        return None


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if all data is complete AND we haven't shown analysis yet
if st.session_state.user_data['data_complete'] and not st.session_state.get('analysis_shown', False):
    
    # Set flag immediately to prevent re-running
    st.session_state.analysis_shown = True
    
    # Store the analyzing message in session state so it persists across reruns
    st.session_state.analyzing_msg = "🔍 Terima kasih! Semua data Anda telah tercatat. Saya akan menganalisis kondisi finansial Anda. dan merujuk ke regulasi OJK..."
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.analyzing_msg})
    
    # Force a rerun to generate the analysis
    st.session_state.should_generate_analysis = True
    st.rerun()

# Generate analysis after rerun
if st.session_state.get('should_generate_analysis', False):
    st.session_state.should_generate_analysis = False  # Reset flag
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare user data for analysis - handle multiple lenders
        user_info = st.session_state.user_data
        total_monthly_installment = sum(lender.get('monthly_installment', 0) for lender in user_info['lenders'])
        total_loan_amount = sum(lender.get('loan_amount', 0) for lender in user_info['lenders'])
        debt_to_income_ratio = (total_monthly_installment / user_info['income']) * 100 if user_info['income'] > 0 else 0
        
        # Build context for all lenders from knowledge base
        lenders_context = ""
        for i, lender in enumerate(user_info['lenders'], 1):
            docs = vectorstore.similarity_search(lender['name'], k=2)
            context_content = "\n".join([doc.page_content for doc in docs])
            lenders_context += f"\n\n**Lender {i} - {lender['name']}:**\n{context_content}"
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""Anda adalah SantAI, analis finansial yang ahli dan empatik. Berikan analisis komprehensif dan rekomendasi berdasarkan data pengguna dan regulasi OJK.

**DATA PENGGUNA:**
- Nama: {user_info['name']}
- Usia: {user_info['age']} tahun
- Pekerjaan: {user_info['job']}
- Penghasilan Bulanan: Rp {user_info['income']:,.0f}
- Total Jumlah Pinjaman: Rp {total_loan_amount:,.0f}
- Total Angsuran Bulanan: Rp {total_monthly_installment:,.0f}
- Rasio Hutang-Penghasilan: {debt_to_income_ratio:.1f}%

**DETAIL LENDER:**
""" + "\n".join([f"- {lender['name']}: Rp {lender.get('loan_amount', 0):,.0f} (Angsuran: Rp {lender.get('monthly_installment', 0):,.0f}/bulan)" 
                for lender in user_info['lenders']]) + f"""

**KONTEKS REGULASI OJK UNTUK SEMUA LENDER:**
{lenders_context}

**TUGAS ANDA:**
1. Analisis kesehatan finansial pengguna berdasarkan rasio hutang-penghasilan total
2. Identifikasi status legalitas masing-masing lender berdasarkan data OJK
3. Berikan rekomendasi spesifik yang sesuai dengan kondisi multi-pinjaman pengguna
4. Prioritaskan lender mana yang harus diselesaikan terlebih dahulu
5. Jelaskan langkah-langkah konkret yang harus diambil
6. Sertakan dasar hukum/referensi regulasi OJK yang relevan
7. Bersikap empatik dan memberikan harapan

**ANALISIS DAN REKOMENDASI:**
"""
        # Generate analysis
        try:
            for chunk in llm.stream(analysis_prompt):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            # Replace the analyzing message with actual analysis using session state
            if (st.session_state.messages and 
                st.session_state.get('analyzing_msg') and 
                st.session_state.messages[-1]["content"] == st.session_state.analyzing_msg):
                st.session_state.messages[-1] = {"role": "assistant", "content": full_response}
            else:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"❌ Maaf, terjadi error dalam menganalisis data: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Show restart button after analysis is shown
if st.session_state.get('analysis_shown', False):
    if st.button("🔄 Mulai Konsultasi Baru", key="restart_btn"):
        # Reset everything
        st.session_state.user_data = {
            'name': None, 'age': None, 'job': None, 'income': None,
            'lender_name': None, 'lender_website': None, 
            'loan_amount': None, 'monthly_installment': None,
            'data_complete': False
        }
        st.session_state.current_step = 'name'
        st.session_state.analysis_shown = False
        st.session_state.should_generate_analysis = False
        st.session_state.analyzing_msg = None  # Clear this too
        st.session_state.messages = []
        st.rerun()

elif llm:  # Only show input if LLM is available
    # Show current question
    current_question = ask_question(st.session_state.current_step)
    if not st.session_state.user_data['data_complete']:
        if not st.session_state.messages or st.session_state.messages[-1]["role"] != "assistant" or st.session_state.messages[-1]["content"] != current_question:
            st.session_state.messages.append({"role": "assistant", "content": current_question})
            with st.chat_message("assistant"):
                st.markdown(current_question)

    # Handle user input
    if prompt := st.chat_input("Jawaban Anda..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the answer based on current step
        current_step = st.session_state.current_step
        
        if current_step == 'name':
            st.session_state.user_data['name'] = prompt
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'age':
            try:
                st.session_state.user_data['age'] = int(prompt)
                st.session_state.current_step = get_next_question()
            except ValueError:
                st.error("Mohon masukkan usia dalam angka")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan usia Anda dalam angka (contoh: 25)"})
                
        elif current_step == 'job':
            st.session_state.user_data['job'] = prompt
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'income':
            try:
                income = extract_number(prompt)
                if income is not None:
                    st.session_state.user_data['income'] = income
                    st.session_state.current_step = get_next_question()
                else:
                    st.error("Mohon masukkan angka yang valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan penghasilan dalam format angka (contoh: 5000000 atau Rp 5.000.000)"})
            except:
                st.error("Mohon masukkan angka yang valid")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan penghasilan dalam format angka (contoh: 5000000 atau Rp 5.000.000)"})
                
        elif current_step == 'lender_name':
            st.session_state.current_lender = {'name': prompt}
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'lender_website':
            st.session_state.current_lender['website'] = prompt if prompt.lower() != 'tidak' else None
            st.session_state.current_step = get_next_question()
            
        elif current_step == 'loan_amount':
            try:
                loan_amount = extract_number(prompt)
                if loan_amount is not None:
                    st.session_state.current_lender['loan_amount'] = loan_amount
                    st.session_state.current_step = get_next_question()
                else:
                    st.error("Mohon masukkan angka yang valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah pinjaman dalam format angka (contoh: 10000000 atau Rp 10.000.000)"})
            except:
                st.error("Mohon masukkan angka yang valid")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah pinjaman dalam format angka (contoh: 10000000 atau Rp 10.000.000)"})
                
        elif current_step == 'monthly_installment':
            try:
                installment = extract_number(prompt)
                if installment is not None:
                    st.session_state.current_lender['monthly_installment'] = installment
                    
                    # Add completed lender to the list
                    st.session_state.user_data['lenders'].append(st.session_state.current_lender.copy())
                    
                    # Ask if user wants to add another lender
                    st.session_state.current_step = 'add_another_lender'
                else:
                    st.error("Mohon masukkan angka yang valid")
                    st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah angsuran dalam format angka (contoh: 1500000 atau Rp 1.500.000)"})
            except:
                st.error("Mohon masukkan angka yang valid")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon masukkan jumlah angsuran dalam format angka (contoh: 1500000 atau Rp 1.500.000)"})
                
        elif current_step == 'add_another_lender':
            if prompt.lower() in ['ya', 'yes', 'y', 'iya', 'iye']:
                st.session_state.current_step = 'next_lender_name'
            elif prompt.lower() in ['tidak', 'no', 'n', 'ga', 'gak']:
                st.session_state.user_data['data_complete'] = True
                st.session_state.current_step = 'complete'
            else:
                st.error("Mohon jawab dengan Ya atau Tidak")
                st.session_state.messages.append({"role": "assistant", "content": "Mohon jawab dengan Ya atau Tidak"})
                
        elif current_step == 'next_lender_name':
            st.session_state.current_lender = {'name': prompt}
            st.session_state.current_step = 'lender_website'

        st.rerun()
