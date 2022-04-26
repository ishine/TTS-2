phonemes = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑ하ᅣᅥᅦᅧᅨᅩᅪᅬᅭᅮᅯᅱᅲᅳᅴᅵᆨᆫᆮᆯᆷᆸᆼ"
pad = "_"
punctuations = " "
symbols = list(pad) + list(punctuations) + list(phonemes)


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    return [_symbol_to_id[s] for s in text]


def sequence_to_text(sequence):
    return "".join([_id_to_symbol[s] for s in sequence])
