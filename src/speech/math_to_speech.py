"""
Math-to-Speech Converter

Converts mathematical notation to spoken French text for TTS.
This allows equations like "x² + 2x - 4 = 0" to be read as
"x au carré plus 2 x moins 4 égale 0"
"""

import re


class MathToSpeech:
    """Converts mathematical expressions to spoken French"""
    
    # Superscript digits mapping
    SUPERSCRIPTS = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        'ⁿ': 'n'
    }
    
    # Subscript digits mapping
    SUBSCRIPTS = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        'ₙ': 'n', 'ₓ': 'x'
    }
    
    # Math symbols to French words
    SYMBOLS = {
        # Basic operators
        '+': ' plus ',
        '-': ' moins ',
        '−': ' moins ',  # Unicode minus
        '×': ' multiplié par ',
        '÷': ' divisé par ',
        '=': ' égale ',
        '≠': ' différent de ',
        '≈': ' environ égal à ',
        
        # Comparisons
        '<': ' inférieur à ',
        '>': ' supérieur à ',
        '≤': ' inférieur ou égal à ',
        '≥': ' supérieur ou égal à ',
        '≪': ' très inférieur à ',
        '≫': ' très supérieur à ',
        
        # Special symbols
        '√': ' racine carrée de ',
        '∛': ' racine cubique de ',
        '∞': ' infini ',
        '∑': ' somme de ',
        '∏': ' produit de ',
        '∫': ' intégrale de ',
        'π': ' pi ',
        'θ': ' thêta ',
        'α': ' alpha ',
        'β': ' bêta ',
        'γ': ' gamma ',
        'δ': ' delta ',
        'Δ': ' delta ',
        'λ': ' lambda ',
        'μ': ' mu ',
        'σ': ' sigma ',
        'ω': ' oméga ',
        
        # Set notation
        '∈': ' appartient à ',
        '∉': ' n\'appartient pas à ',
        '⊂': ' inclus dans ',
        '∪': ' union ',
        '∩': ' intersection ',
        '∅': ' ensemble vide ',
        
        # Logic
        '∀': ' pour tout ',
        '∃': ' il existe ',
        '¬': ' non ',
        '∧': ' et ',
        '∨': ' ou ',
        '⇒': ' implique ',
        '⇔': ' équivaut à ',
        
        # Arrows
        '→': ' tend vers ',
        '←': ' ',
        '↔': ' ',
        
        # Other
        '°': ' degrés ',
        '%': ' pourcent ',
        '±': ' plus ou moins ',
        '∓': ' moins ou plus ',
    }
    
    # Common fractions
    FRACTIONS = {
        '½': 'un demi',
        '⅓': 'un tiers',
        '⅔': 'deux tiers',
        '¼': 'un quart',
        '¾': 'trois quarts',
        '⅕': 'un cinquième',
        '⅖': 'deux cinquièmes',
        '⅗': 'trois cinquièmes',
        '⅘': 'quatre cinquièmes',
        '⅙': 'un sixième',
        '⅚': 'cinq sixièmes',
        '⅛': 'un huitième',
        '⅜': 'trois huitièmes',
        '⅝': 'cinq huitièmes',
        '⅞': 'sept huitièmes',
    }
    
    def __init__(self):
        pass
    
    def convert(self, text: str) -> str:
        """
        Convert mathematical notation in text to spoken French.
        
        Args:
            text: Text potentially containing math symbols
            
        Returns:
            Text with math converted to spoken words
        """
        result = text
        
        # 1. Handle fractions first
        for frac, spoken in self.FRACTIONS.items():
            result = result.replace(frac, f' {spoken} ')
        
        # 2. Handle superscripts (powers)
        result = self._convert_powers(result)
        
        # 3. Handle subscripts
        result = self._convert_subscripts(result)
        
        # 4. Handle x^2, x^3 notation
        result = self._convert_caret_powers(result)
        
        # 5. Handle division with /
        result = self._convert_fractions_slash(result)
        
        # 6. Replace math symbols
        for symbol, spoken in self.SYMBOLS.items():
            result = result.replace(symbol, spoken)
        
        # 7. Handle multiplication with *
        result = result.replace('*', ' multiplié par ')
        
        # 8. Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        # 9. Add pauses for better rhythm
        result = self._add_pauses(result)
        
        return result.strip()
    
    def _convert_powers(self, text: str) -> str:
        """Convert superscript numbers to 'au carré', 'au cube', etc."""
        result = text
        
        # Find sequences of superscripts
        superscript_pattern = r'([a-zA-Z0-9])([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]+)'
        
        def replace_power(match):
            base = match.group(1)
            power_chars = match.group(2)
            
            # Convert superscript to normal digits
            power = ''.join(self.SUPERSCRIPTS.get(c, c) for c in power_chars)
            
            if power == '2':
                return f'{base} au carré'
            elif power == '3':
                return f'{base} au cube'
            elif power == 'n':
                return f'{base} puissance n'
            else:
                return f'{base} puissance {power}'
        
        result = re.sub(superscript_pattern, replace_power, result)
        return result
    
    def _convert_subscripts(self, text: str) -> str:
        """Convert subscript numbers to 'indice X'"""
        result = text
        
        subscript_pattern = r'([a-zA-Z])([₀₁₂₃₄₅₆₇₈₉ₙₓ]+)'
        
        def replace_subscript(match):
            base = match.group(1)
            sub_chars = match.group(2)
            sub = ''.join(self.SUBSCRIPTS.get(c, c) for c in sub_chars)
            return f'{base} indice {sub}'
        
        result = re.sub(subscript_pattern, replace_subscript, result)
        return result
    
    def _convert_caret_powers(self, text: str) -> str:
        """Convert x^2, x^3 notation"""
        result = text
        
        # Pattern: letter or number followed by ^ and number
        caret_pattern = r'([a-zA-Z0-9])\^(\d+|n)'
        
        def replace_caret(match):
            base = match.group(1)
            power = match.group(2)
            
            if power == '2':
                return f'{base} au carré'
            elif power == '3':
                return f'{base} au cube'
            elif power == 'n':
                return f'{base} puissance n'
            else:
                return f'{base} puissance {power}'
        
        result = re.sub(caret_pattern, replace_caret, result)
        return result
    
    def _convert_fractions_slash(self, text: str) -> str:
        """Convert a/b to 'a sur b' or 'a divisé par b'"""
        result = text
        
        # Pattern: number/number or (expr)/(expr)
        fraction_pattern = r'(\d+)\s*/\s*(\d+)'
        
        def replace_fraction(match):
            num = match.group(1)
            den = match.group(2)
            
            # Common fractions
            if num == '1' and den == '2':
                return 'un demi'
            elif num == '1' and den == '3':
                return 'un tiers'
            elif num == '1' and den == '4':
                return 'un quart'
            else:
                return f'{num} sur {den}'
        
        result = re.sub(fraction_pattern, replace_fraction, result)
        return result
    
    def _add_pauses(self, text: str) -> str:
        """Add natural pauses for better TTS rhythm"""
        result = text
        
        # Add slight pause after equals
        result = result.replace(' égale ', ' égale, ')
        
        # Add pause after "donc"
        result = result.replace(' donc ', ' donc, ')
        
        # Ensure pause after periods
        result = re.sub(r'\.(?!\s)', '. ', result)
        
        # Add pause before "où" (in equations)
        result = result.replace(' où ', ', où ')
        
        return result


# Singleton instance for easy import
math_to_speech = MathToSpeech()


def convert_math_to_speech(text: str) -> str:
    """
    Convenience function to convert math in text to spoken French.
    
    Usage:
        from math_to_speech import convert_math_to_speech
        spoken = convert_math_to_speech("x² + 2x - 4 = 0")
        # Returns: "x au carré plus 2 x moins 4 égale 0"
    """
    return math_to_speech.convert(text)


if __name__ == "__main__":
    # Test examples
    tests = [
        "x² + 2x - 4 = 0",
        "La formule est E = mc²",
        "√16 = 4",
        "π ≈ 3.14159",
        "x³ - 8 = 0",
        "1/2 + 1/4 = 3/4",
        "∫f(x)dx",
        "∑(i=1 to n)",
        "x₁ + x₂ = 10",
        "a^2 + b^2 = c^2",
    ]
    
    converter = MathToSpeech()
    for test in tests:
        print(f"Original: {test}")
        print(f"Spoken:   {converter.convert(test)}")
        print()
