def generate_symbol_to_id(symbols):
    return  {s: i for i, s in enumerate(symbols)}


def generate_id_to_symbol(symbols):
    return {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, symbol_to_id):
    return [symbol_to_id[s] for s in text]


def sequence_to_text(sequence, id_to_symbol):
    return "".join([id_to_symbol[s] for s in sequence])
