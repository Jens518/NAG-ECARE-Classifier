import pandas as pd
import re
import spacy
from collections import Counter

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")  # Use smaller model

class ECAREClassifier:
    def __init__(self, taxonomy_file):
        """Initialize the classifier with the ECARE taxonomy Excel file."""
        self.taxonomy_df = pd.read_excel(taxonomy_file)
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
                
            # Process the description with spaCy
            doc = nlp(description.lower())
            
            # Extract relevant keywords (nouns, verbs, adjectives)
            keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3]
            
            # Add each keyword to the index
            for keyword in keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(code)

        # Add debug print
        print(f"Keyword index created with {len(keyword_index)} keywords")
        
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
        # Process the input text with spaCy
        doc = nlp(text.lower())
        
        # Extract relevant keywords
        input_keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3]
        
        # Count matching codes for each keyword
        code_matches = Counter()
        
        for keyword in input_keywords:
            if keyword in self.keyword_index:
                for code in self.keyword_index[keyword]:
                    code_matches[code] += 1
        
        # Prioritize more specific codes
        for code, count in list(code_matches.items()):
            # If a specific code (e.g., A1.01) is matched, also boost its parent codes (A1., A.)
            if re.match(r'^[A-Z][0-9]+\.[0-9]+$', code):  # Third level
                parent1 = code.split(".")[0] + "."
                parent2 = code[0] + "."
                code_matches[parent1] += count * 0.5
                code_matches[parent2] += count * 0.25
            elif re.match(r'^[A-Z][0-9]+\.$', code):  # Second level
                parent = code[0] + "."
                code_matches[parent] += count * 0.5
        
        # Get top N codes
        top_codes = [code for code, _ in code_matches.most_common(top_n)]
        
        # Simplify by removing parent codes if all children are included
        simplified_codes = self._simplify_code_list(top_codes)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(simplified_codes, text)
        
        return {
            "codes": simplified_codes,
            "reasoning": reasoning
        }
        
        # Debug prints
        print(f"Input text: {text[:100]}...")
        print(f"Extracted keywords: {input_keywords[:20]}")
        print(f"Matching keywords: {[k for k in input_keywords if k in self.keyword_index]}")
    
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
                doc1 = nlp(description.lower())
                doc2 = nlp(text.lower())
                
                # Find similarity between code description and input text
                similarity = doc1.similarity(doc2)
                
                # Extract matching keywords
                desc_keywords = set([token.lemma_ for token in doc1 if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3])
                text_keywords = set([token.lemma_ for token in doc2 if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 3])
                matching_keywords = desc_keywords.intersection(text_keywords)
                
                # Generate reasoning
                reasoning[code] = f"{code} ({description}): Matches keywords [{', '.join(list(matching_keywords)[:5])}]"
        
        return reasoning
