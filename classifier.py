import pandas as pd
import re
import spacy
from collections import Counter

# Load spaCy model for NLP - use the smaller model for better performance
nlp = spacy.load("en_core_web_sm")

class ECAREClassifier:
    def __init__(self, taxonomy_file):
        """Initialize the classifier with the ECARE taxonomy Excel file."""
        print(f"Loading taxonomy from: {taxonomy_file}")
        try:
            self.taxonomy_df = pd.read_excel(taxonomy_file)
            print(f"Successfully loaded taxonomy with {len(self.taxonomy_df)} entries")
            # Print first few rows to verify content
            print("First few taxonomy entries:")
            print(self.taxonomy_df.head())
        except Exception as e:
            print(f"Error loading taxonomy file: {e}")
            raise
            
        self.taxonomy_dict = self._create_taxonomy_dict()
        self.keyword_index = self._create_keyword_index()
        self.parent_codes = self._identify_parent_codes()
    
    def _create_taxonomy_dict(self):
        """Create a dictionary mapping taxonomy codes to their descriptions."""
        return dict(zip(self.taxonomy_df["Taxonomy"], self.taxonomy_df["Description"]))
    
    def _create_keyword_index(self):
        """Create an index of keywords to taxonomy codes."""
        keyword_index = {}
        
        for _, row in self.taxonomy_df.iterrows():
            code = row["Taxonomy"]
            description = row["Description"]
            
            # Skip empty descriptions
            if not isinstance(description, str):
                continue
                
            # Add the description words directly
            keywords = [word.lower() for word in description.split() if len(word) > 3]
            
            # Also use spaCy for better keyword extraction
            doc = nlp(description.lower())
            keywords.extend([token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3])
            
            # Make keywords unique
            keywords = list(set(keywords))
            
            # Add each keyword to the index
            for keyword in keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                if code not in keyword_index[keyword]:
                    keyword_index[keyword].append(code)
        
        print(f"Created keyword index with {len(keyword_index)} keywords")
        # Print a sample of keywords
        sample_keywords = list(keyword_index.keys())[:10]
        for keyword in sample_keywords:
            print(f"Keyword '{keyword}' maps to: {keyword_index[keyword]}")
            
        return keyword_index
    
    def _identify_parent_codes(self):
        """Identify parent codes in the taxonomy hierarchy."""
        parent_codes = {}
        
        for code in self.taxonomy_dict.keys():
            # Match top-level codes (e.g., "A.")
            if re.match(r'^[A-Z]\.$', code):
                parent_codes[code] = []
            # Match second-level codes (e.g., "A1.")
            elif re.match(r'^[A-Z][0-9]+\.$', code):
                parent = code[0] + "."
                if parent in parent_codes:
                    parent_codes[parent].append(code)
            # Match third-level codes (e.g., "A1.01")
            elif re.match(r'^[A-Z][0-9]+\.[0-9]+$', code):
                parent = code.split(".")[0] + "."
                if parent in self.taxonomy_dict:
                    if parent not in parent_codes:
                        parent_codes[parent] = []
                    parent_codes[parent].append(code)
        
        return parent_codes
    
    def classify_text(self, text, top_n=15):
        """Classify text according to the ECARE taxonomy."""
        print(f"Classifying text: {text[:100]}...")
        
        # Direct keyword matching - split the text into words
        direct_keywords = [word.lower() for word in text.split() if len(word) > 3]
        
        # Also process with spaCy for better keyword extraction
        doc = nlp(text.lower())
        spacy_keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3]
        
        # Combine both methods
        input_keywords = list(set(direct_keywords + spacy_keywords))
        
        print(f"Extracted {len(input_keywords)} keywords: {input_keywords[:10]}...")
        
        # Count matching codes for each keyword
        code_matches = Counter()
        matching_keywords = []
        
        for keyword in input_keywords:
            if keyword in self.keyword_index:
                matching_keywords.append(keyword)
                for code in self.keyword_index[keyword]:
                    code_matches[code] += 1
        
        print(f"Found {len(matching_keywords)} matching keywords: {matching_keywords}")
        
        # Look for multi-word phrases in the taxonomy descriptions
        for code, description in self.taxonomy_dict.items():
            description_lower = description.lower()
            if any(phrase.lower() in description_lower for phrase in [
                "additive manufacturing", "concurrent engineering", "manufacturing process",
                "knowledge engineering", "flight physics", "systems engineering"
            ]):
                for phrase in [
                    "additive manufacturing", "concurrent engineering", "manufacturing process",
                    "knowledge engineering", "flight physics", "systems engineering"
                ]:
                    if phrase.lower() in description_lower and phrase.lower() in text.lower():
                        print(f"Found exact phrase match: '{phrase}' in code {code}")
                        code_matches[code] += 5  # Give higher weight to exact phrase matches
        
        # Check for exact taxonomy description matches
        for code, description in self.taxonomy_dict.items():
            if description.lower() in text.lower():
                print(f"Found exact description match: '{description}' for code {code}")
                code_matches[code] += 10  # Give highest weight to exact description matches
        
        print(f"Found {len(code_matches)} matching codes")
        
        # Get top N codes
        top_codes = [code for code, _ in code_matches.most_common(top_n)]
        
        if not top_codes:
            # If no matches, check for partial matches in code descriptions
            for code, description in self.taxonomy_dict.items():
                desc_lower = description.lower()
                text_lower = text.lower()
                
                # Check if any word from the text (with length > 3) is in the description
                text_words = [word for word in text_lower.split() if len(word) > 3]
                desc_words = [word for word in desc_lower.split() if len(word) > 3]
                
                for word in text_words:
                    if word in desc_words:
                        print(f"Found partial word match: '{word}' in code {code}")
                        code_matches[code] += 1
            
            # Try again with the updated code_matches
            top_codes = [code for code, _ in code_matches.most_common(top_n)]
        
        # Simplify by removing parent codes if all children are included
        simplified_codes = self._simplify_code_list(top_codes)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(simplified_codes, text)
        
        if not simplified_codes:
            simplified_codes = ["No matching codes found"]
        
        return {
            "codes": simplified_codes,
            "reasoning": reasoning
        }
    
    def _simplify_code_list(self, codes):
        """Simplify the code list by consolidating parent codes."""
        simplified = codes.copy()
        
        # Check for parent codes where all children are present
        for parent, children in self.parent_codes.items():
            if parent in simplified and all(child in simplified for child in children if child in self.taxonomy_dict):
                # Remove children and keep only the parent
                for child in children:
                    if child in simplified:
                        simplified.remove(child)
        
        return simplified
    
    def _generate_reasoning(self, codes, text):
        """Generate reasoning for why each code was selected."""
        reasoning = {}
        
        for code in codes:
            if code in self.taxonomy_dict:
                description = self.taxonomy_dict[code]
                
                # Find matching keywords
                desc_keywords = set([word.lower() for word in description.split() if len(word) > 3])
                text_keywords = set([word.lower() for word in text.split() if len(word) > 3])
                matching_keywords = desc_keywords.intersection(text_keywords)
                
                # If no direct keyword matches, try using spaCy
                if not matching_keywords:
                    doc1 = nlp(description.lower())
                    doc2 = nlp(text.lower())
                    
                    desc_keywords = set([token.lemma_ for token in doc1 if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3])
                    text_keywords = set([token.lemma_ for token in doc2 if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3])
                    matching_keywords = desc_keywords.intersection(text_keywords)
                
                # Generate reasoning
                if matching_keywords:
                    reasoning[code] = f"{code} ({description}): Matches keywords [{', '.join(list(matching_keywords)[:5])}]"
                else:
                    reasoning[code] = f"{code} ({description}): Semantic match with the text content"
        
        return reasoning
