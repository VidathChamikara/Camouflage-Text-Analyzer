import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
from annotated_text import annotated_text
from better_profanity import profanity
import plotly.express as px
from collections import Counter

# ===================================================================
# 1. Configuration & Page Setup
# ===================================================================

st.set_page_config(
    page_title="Camouflage Text Analyzer",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stPlotlyChart { .modebar { display: none !important; } }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "xlm-roberta-large-ner-model-emoji"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# 2. Load Model and Tokenizer (with Caching)
# ===================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """Loads and caches the NER model, tokenizer, and id_to_label mapping."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        id_to_label = model.config.id2label
        return tokenizer, model, id_to_label
    except OSError:
        st.error(f"Error loading model from path: {MODEL_PATH}. Please ensure the model files are in the correct directory.")
        return None, None, None

# ===================================================================
# 3. Analysis Helpers (ENHANCED WITH EMOJI MAPPING)
# ===================================================================

WORD_TO_EMOJI_MAP = {
    "shit": ["💩", "🤢"], "fuck": ["🖕", "🍆"], "fucking": ["🖕"], "bitch": ["🐶", "🐩", "💅"], "ass": ["🍑", "🍩"],
    "asshole": ["🍑", "🚽"], "trash": ["🗑️", "🚮"], "jerk": ["🦍"], "clown": ["🤡"], "dumb": ["🤪", "🙃"],
    "loser": ["🦆", "🙅‍♂️"], "gay": ["🏳️‍🌈"], "monkey": ["🐒", "🐵"], "rat": ["🐀", "🐭"], "snake": ["🐍"],
    "pig": ["🐷", "🐖"], "cow": ["🐄", "🐮"], "donkey": ["🐴"], "chicken": ["🐔", "🐤"], "dog": ["🐶", "🐕"],
    "whore": ["💋", "💃"], "slut": ["👠", "💃"], "idiot": ["🦆", "🤡"], "nerd": ["🤓"], "nigga": ["🐵", "🐒"],
    "nigger": ["🐵", "🐒"], "niggah": ["🐵", "🐒"], "coon": ["🦝"], "chink": ["🥢", "👲"], "kike": ["✡️"],
    "spic": ["🌮"], "tranny": ["👩‍🎤"], "dyke": ["🌈"], "dick": ["🍆", "🍌", "🥒"], "cock": ["🐓", "🐔"],
    "pussy": ["🐱", "🐈", "🌮", "🌸"], "tits": ["🍒", "🍈"], "boobs": ["🍈", "🎈"], "balls": ["⚽", "🏀", "🎱"],
    "cum": ["💦", "💧"], "anus": ["🍑", "🕳️"], "fart": ["💨"], "kill": ["🔪", "🔫", "☠️"], "murder": ["🔪", "☠️"],
    "hate": ["😡", "😠", "💔"], "sad": ["😢", "😭"], "cry": ["😭"], "love": ["❤️", "😍", "💖"], "fire": ["🔥"],
    "hot": ["🔥"], "cold": ["❄️"], "money": ["💰", "💵", "🤑"], "rich": ["💸", "🤑"], "poor": ["💸", "😢"],
    "weed": ["🌿", "🍁", "💨"], "pot": ["🍁"], "heroin": ["💉"], "meth": ["💊"], "cocaine": ["❄️"],
    "alcohol": ["🍺", "🍻", "🥃"], "brain": ["🧠"], "eyes": ["👀"], "skull": ["💀"], "cap": ["🧢"],
    "star": ["⭐"], "devil": ["😈"], "angel": ["😇"], "karen": ["🙎‍♀️"], "simp": ["🤦‍♂️", "🙇‍♂️"],
    "thot": ["💃", "👠"], "chad": ["💪"], "stacy": ["💃"]
}
CHAR_TO_EMOJI_MAP = {
    'a': ['@', '🅰️', '🔼', 'Λ', '4', 'α', '∂', 'Ʌ'], 'b': ['🅱️', '₿', 'β', '8', '13', 'ß'],
    'c': ['©', '🅒', 'ↄ', '⊂', '('], 'd': ['ↁ', '∂', 'ԁ', 'đ'], 'e': ['3', '📧', '€', 'ε', 'ɘ', '℮'],
    'f': ['🎏', 'ƒ', '₣'], 'g': ['6', '9', 'ɢ', '🌀'], 'h': ['#', '♓', 'ħ', '|-|'],
    'i': ['ℹ️', '🇮', '1', '❗', '¡', '|', '丨', 'ɪ'], 'j': ['¿', 'ĵ', 'ʝ'], 'k': ['ⲕ', 'Ҝ', 'κ', '|<'],
    'l': ['1', '|', 'ℓ', '🦵'], 'm': ['♏', 'м', '₥', '/\\/\\'], 'n': ['♑', 'и', 'ท', '₪', '|\\|'],
    'o': ['⭕', '🅾️', '0️⃣', '0', 'ο', 'θ', '⦿'], 'p': ['🅿️', 'ρ', '¶', '|*'], 'q': ['9', '🍳', 'ϙ', 'ԛ'],
    'r': ['®️', 'Я', '₹', 'ɾ'], 's': ['💲', '⚡', '💰', '$', '5', '§'], 't': ['✝️', '➕', '7', '+', '†', '☦'],
    'u': ['∪', 'υ', 'µ', 'บ', '[_]'], 'v': ['✔️', '√', '\\/', 'ν', '∨'], 'w': ['Ⱳ', 'ω', '\\/\\/', 'Ш'],
    'x': ['❌', '❎', '✖️', '×', '✗', '><'], 'y': ['¥', 'γ', 'ɣ', 'ʏ'], 'z': ['2', 'Ⓩ', 'Ⱬ', 'ʐ']
}

EMOJI_TO_WORD_MAP = {emoji: word for word, emojis in WORD_TO_EMOJI_MAP.items() for emoji in emojis}
EMOJI_CHAR_TO_CHAR_MAP = {emoji: char for char, emojis in CHAR_TO_EMOJI_MAP.items() for emoji in emojis}
MULTI_CHAR_MAP = {'\\/\\/': 'w', '\\/': 'v', '|3': 'b', '|o': 'o', '|\\|': 'n', 'ph': 'f', 'ck': 'k'}
SINGLE_CHAR_MAP = {
    '0': 'o', '1': 'l', '2': 'z', '3': 'e', '4': 'a', '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
    '@': 'a', '$': 's', '+': 't', '!': 'i', '|': 'l', '?': 'q', '#': 'h', '&': 'a', '%': 'x', '^': 'v',
    '(': 'c', ')': 'c', '{': 'c', '}': 'c', '[': 'c', ']': 'c', '*': '', '/': '', '\\': '', '-': '',
    '_': '', '.': '', ',': '', '~': '',
}

def deobfuscate_entity(entity_text, label):
    if 'EMOJI_CAMO' in label:
        return "".join([EMOJI_TO_WORD_MAP.get(char, char) for char in entity_text])
    else:
        text = "".join([EMOJI_CHAR_TO_CHAR_MAP.get(char, char) for char in entity_text])
        text = text.lower()
        for pattern, replacement in MULTI_CHAR_MAP.items():
            text = text.replace(pattern, replacement)
        final_chars = [SINGLE_CHAR_MAP.get(char, char) for char in text]
        return "".join(final_chars)

def classify_offensiveness(deobfuscated_text):
    return "Offensive" if profanity.contains_profanity(deobfuscated_text) else "Not Offensive"

# ===================================================================
# 3.5. REVISED: User-Friendly Explanation Generator
# ===================================================================

def generate_specific_explanation(entity_text, deobfuscated_text, label):
    """
    Generates a clear, user-friendly explanation of how a single entity was camouflaged.
    """
    # Case 1: Semantic Emoji Swap (e.g., 💩 -> shit)
    if label == "EMOJI_CAMO":
        return f"The emoji '{entity_text}' was used to represent the word '{deobfuscated_text}'."

    # Case 2: Visual Emoji Character Swap (e.g., l❤️ve -> love)
    if label == "EMOJI_CHAR_CAMO":
        swaps = []
        for char in entity_text:
            if char in EMOJI_CHAR_TO_CHAR_MAP:
                swaps.append(f"the letter '{EMOJI_CHAR_TO_CHAR_MAP[char]}' with the emoji '{char}'")
        if swaps:
            return f"The word '{deobfuscated_text}' was camouflaged by replacing {', '.join(swaps)}."
        else: # Fallback
            return f"An emoji was used to visually camouflage the word '{deobfuscated_text}'."

    # Case 3: Character-based camouflage (Leetspeak, Punctuation, Mixed)
    substitutions = []
    insertions = []
    
    temp_entity_text = entity_text
    
    # Identify characters that were simply inserted and then removed (like hyphens)
    for char in entity_text:
        if SINGLE_CHAR_MAP.get(char) == '':
            insertions.append(f"'{char}'")
            temp_entity_text = temp_entity_text.replace(char, '')

    # Identify character substitutions
    if len(temp_entity_text) == len(deobfuscated_text):
        for original_char, deob_char in zip(temp_entity_text, deobfuscated_text):
            if original_char.lower() != deob_char.lower():
                substitutions.append(f"the letter '{deob_char}' with '{original_char}'")
    
    # --- Build the final, user-friendly explanation ---
    explanation_clauses = []
    if substitutions:
        # Sort to ensure consistent output order
        unique_subs = sorted(list(set(substitutions)))
        if len(unique_subs) == 1:
            clause = "replacing " + unique_subs[0]
        else:
            # Join multiple substitutions for a clean list, e.g., "...replacing the letter 'e' with '3' and the letter 'o' with '0'"
            clause = "replacing " + " and ".join(unique_subs)
        explanation_clauses.append(clause)

    if insertions:
        unique_ins = sorted(list(set(insertions)))
        if len(unique_ins) == 1:
            clause = "inserting the character " + unique_ins[0]
        else:
            # e.g., "...inserting the characters '-' and '_'"
            clause = "inserting characters like " + " and ".join(unique_ins)
        explanation_clauses.append(clause)

    if explanation_clauses:
        # Combine all parts into a final sentence.
        return f"The word '{deobfuscated_text}' was camouflaged by {' and '.join(explanation_clauses)}."
    
    # Fallback for any case not caught
    return f"A general camouflage technique was used to hide the word '{deobfuscated_text}'."

def refine_camouflage_label(entity_text, original_label):
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\u2600-\u26FF\u2700-\u27BF]+",
        flags=re.UNICODE)
    has_digits = bool(re.search(r"\d", entity_text))
    has_punct = bool(re.search(r"[^\w\s\d]", entity_text))
    has_emoji = bool(emoji_pattern.search(entity_text))

    if has_emoji:
        return "EMOJI_CAMO" if entity_text in EMOJI_TO_WORD_MAP and len(entity_text) == 1 else "EMOJI_CHAR_CAMO"
    if has_digits and not has_punct: return "LEETSPEAK"
    if has_punct and not has_digits: return "PUNCT_CAMO"
    if has_digits and has_punct: return "MIX_CAMO"
    return original_label

# ===================================================================
# 4. Prediction function with full analysis pipeline
# ===================================================================

def predict_entities(text, model, tokenizer, id_to_label, device):
    if not text.strip(): return []
    inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, axis=2).squeeze().tolist()
    entities, current_entity = [], None
    for i, pred_id in enumerate(predicted_ids):
        label = id_to_label[pred_id]
        char_start, char_end = offset_mapping[i]
        if char_start == char_end: continue
        if label.startswith("B-"):
            if current_entity: entities.append(current_entity)
            current_entity = {"entity": text[char_start:char_end], "label": label[2:], "start": int(char_start), "end": int(char_end)}
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["label"]:
            current_entity["end"] = int(char_end)
            current_entity["entity"] = text[current_entity["start"]:current_entity["end"]]
        else:
            if current_entity: entities.append(current_entity)
            current_entity = None
    if current_entity: entities.append(current_entity)

    for entity in entities:
        entity["label"] = refine_camouflage_label(entity["entity"], entity["label"])
        entity['deobfuscated'] = deobfuscate_entity(entity['entity'], entity["label"])
        entity['offensiveness'] = classify_offensiveness(entity['deobfuscated'])
        entity['explanation'] = generate_specific_explanation(
            entity['entity'],
            entity['deobfuscated'],
            entity['label']
        )
    return entities

# ===================================================================
# 5. Helper function for displaying results
# ===================================================================

def display_annotated_text(text, entities):
    if not entities:
        st.write(text)
        return
    colors = {"Offensive": "#FF4B4B", "Not Offensive": "#3D85C6"}
    entities.sort(key=lambda x: x['start'])
    annotated_parts = []
    last_end = 0
    for entity in entities:
        if entity['start'] > last_end:
            annotated_parts.append(text[last_end:entity['start']])
        bg_color = colors.get(entity['offensiveness'], "#D3D3D3")
        annotated_parts.append((entity['entity'], f"{entity['label']}", bg_color))
        last_end = entity['end']
    if last_end < len(text):
        annotated_parts.append(text[last_end:])
    annotated_text(*annotated_parts)

# ===================================================================
# 6. Streamlit User Interface
# ===================================================================

st.title("🕵️ Camouflage Text Analyzer")
st.markdown("This tool uses a fine-tuned `XLM-RoBERTa-Large` model to detect obfuscated text, translate it, and classify its potential intent.")

with st.spinner(f"Loading model on {DEVICE}... (This may take a moment on first run)"):
    tokenizer, model, id_to_label = load_model_and_tokenizer()

st.sidebar.header("⚙️ Options & Examples")
st.sidebar.info(f"Using device: **{str(DEVICE).upper()}**")

examples = {
    "No Obfuscation": "This is a normal sentence without any hidden meaning.",
    "Leetspeak": "Th1s |s my n3w \\/\\/ebsite, ch3ck it out!",    
    "Punctuation Camo": "I want to buy some p-i-l-l-s.",
    "Mixed Camo": "G3t y0ur fr33 💰 n0w!!",
    "Emoji Char Camo": "That is some bu||$♓|t!",
    "Emoji Camo": "Let's go to the 🏖️ and have 🍕. Don't be a 🤡.",
    "Combined Example": "Th1s is a t3st. I l❤️ve streamlit! Buy our pr0ducts n0w!!! C@ll us.",
    "Offensive": "What the h3ll is this 💩? Buy v|agra now at our c@sino!",
}
selected_example = st.sidebar.selectbox(
    "Choose an example to analyze:",
    options=list(examples.keys()),
    index=7
)
example_text = examples[selected_example]

st.subheader("Enter Text to Analyze")
user_input = st.text_area(
    "Or type your own text here:",
    value=example_text,
    height=150,
    key="text_area"
)

if st.button("🔍 Analyze Text"):
    if model and tokenizer and user_input:
        with st.spinner("Analyzing..."):
            detected_entities = predict_entities(user_input, model, tokenizer, id_to_label, DEVICE)

        st.subheader("📊 Results")

        total_count = len(detected_entities)
        offensive_count = sum(1 for e in detected_entities if e['offensiveness'] == "Offensive")
        not_offensive_count = total_count - offensive_count

        if st.session_state.get('has_run_before', False):
            delta_total = total_count - st.session_state.get('prev_total', 0)
            delta_offensive = offensive_count - st.session_state.get('prev_offensive', 0)
            delta_not_offensive = not_offensive_count - st.session_state.get('prev_not_offensive', 0)
        else:
            delta_total, delta_offensive, delta_not_offensive = None, None, None

        st.markdown("**Analysis Summary**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Entities Found", value=total_count, delta=delta_total)
        with col2:
            st.metric(label="Offensive/Spam Entities", value=offensive_count, delta=delta_offensive, delta_color="inverse")
        with col3:
            st.metric(label="Not Offensive Entities", value=not_offensive_count, delta=delta_not_offensive)

        st.session_state.prev_total = total_count
        st.session_state.prev_offensive = offensive_count
        st.session_state.prev_not_offensive = not_offensive_count
        st.session_state.has_run_before = True

        st.markdown("---")

        if detected_entities:
            tab1, tab2 = st.tabs(["📄 Detailed Analysis", "📈 Visual Breakdown"])

            with tab1:
                st.markdown("**Annotated Text** (Red for Offensive/Spam, Blue for Not Offensive):")
                display_annotated_text(user_input, detected_entities)
                st.markdown("---")
                st.markdown("**Detected Entities Table:**")
                df = pd.DataFrame(detected_entities)
                
                df.rename(columns={'explanation': 'Tactic Explanation'}, inplace=True)
                
                st.dataframe(
                    df[['entity', 'label', 'deobfuscated', 'Tactic Explanation', 'offensiveness']],
                    use_container_width=True
                )

            with tab2:
                st.markdown("**Result Distribution**")
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    offensiveness_counts = Counter(e['offensiveness'] for e in detected_entities)
                    offensiveness_df = pd.DataFrame(offensiveness_counts.items(), columns=['Category', 'Count'])
                    fig1 = px.pie(offensiveness_df, names='Category', values='Count', title='Intent Breakdown',
                                 color='Category', color_discrete_map={'Offensive': '#FF4B4B', 'Not Offensive': '#3D85C6'})
                    fig1.update_layout(legend_title_text='Intent')
                    st.plotly_chart(fig1, use_container_width=True)

                with chart_col2:
                    label_counts = Counter(e['label'] for e in detected_entities)
                    label_df = pd.DataFrame(label_counts.items(), columns=['Camouflage Type', 'Count'])
                    fig2 = px.pie(label_df, names='Camouflage Type', values='Count', title='Camouflage Type Breakdown',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig2.update_layout(legend_title_text='Type')
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("✅ No obfuscated entities were detected in the text.")

    elif not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        st.error("Model not loaded. Cannot perform analysis.")