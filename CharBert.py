# character bert

import itertools, math, os, random, torch
import torch.nn as nn
import torch.nn.functional as F

def hyp():  # hyperparameters
    global batSiz, batTim, drpRat, embDim, heaSiz, mskRat, numLyr, splRat, numHea  # parameters for transformer
    batSiz = 4  # bat-size
    batTim = 253  # bat-time (odd number ensured)
    drpRat = 0.2  # dropout-rate
    embDim = 128  # embedding-dimension
    heaSiz = 16  # head-size
    mskRat = 0.15  # masking-rate of a token
    numLyr = 8  # number-of-layers
    splRat = 0.8  # (training-validation) splitting-rate
    numHea = embDim // heaSiz  # number-of-heads

    global itrLss, itrOpt, stpLen, obsOpt  # parameters for training
    itrLss = [5, 5]  # iteration-times-for-loss-estimation in [pretrain, fine-tune]
    itrOpt = [10, 10, 10]  # iteration-times-for-optimization in [pretrain, full-fine-tune, lora-fine-tune]
    stpLen = [3e-4, 2e-4, 1e-4]  # step-txtPreLen for [pretrain, full-fine-tune, lora-fine-tune]
    obsOpt = [2, 2, 2]  # observation of loss after each n steps for [pretrain, full-fine-tune, lora-fine-tune]

    global lora, lraAlp, lraDrp, lraRnk, lraScl  # parameters for lora (low-rank-adaptation)
    lora = 0.0  # lora switch initialization. 0 to turn off lora, and 1 to turn on
    lraAlp = 16  # lora-alpha
    lraDrp = 0.1  # lora-dropout
    lraRnk = 8  # lora-rank
    lraScl = lraAlp / lraRnk  # lora-scale


def setup():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    # load pretrain text
    global preTrnTxt, preValTxt, preTrnTxtLen, preValTxtLen, vocabulary
    with open('text_pretrain.txt', encoding="utf-8") as file:
        preTxt = file.read()  # pretrain-text

        preTxtLen = len(preTxt)  # count characters
        splTim = int(preTxtLen * splRat)
        preTrnTxt = preTxt[:splTim]
        preValTxt = preTxt[splTim:]
        preTrnTxtLen = splTim
        preValTxtLen = len(preValTxt)

        assert (preTrnTxtLen >= batTim - 1)
        assert (preValTxtLen >= batTim - 1)
        vocabulary = set(preTxt)

    # load theme: predict theme by given text
    global thmTrnCls, thmValCls, thmTrnTxt, thmValTxt, thmTrnSiz, thmValSiz, numThm
    with open('text_theme.txt',
              encoding="utf-8") as file:  # each line is of 400 ~ 500 characters
        thmCls, thmTxt = [], []
        for line in file:
            line = line.strip()  # remove trailing newline / spaces
            if not line:  # skip empty lines
                continue
            lines = line.split("\t")  # split by tab
            if len(lines) != 2:  # safety check
                continue
            theme, sentence = lines
            thmCls.append(theme)
            thmTxt.append(sentence)

        themes = sorted(set(thmCls))
        numThm = len(themes)
        thm2idx = {c: i for i, c in enumerate(themes)}
        thmCls = [thm2idx[t] for t in thmCls]
        idx2thm = {i: c for i, c in enumerate(themes)}

        thmTrnSiz = int(len(thmCls) * splRat)
        thmTrnCls, thmValCls = thmCls[:thmTrnSiz], thmCls[thmTrnSiz:]
        thmTrnTxt, thmValTxt = thmTxt[:thmTrnSiz], thmTxt[thmTrnSiz:]
        thmValSiz = len(thmValCls)

        assert (thmTrnSiz >= batSiz)
        assert (thmValSiz >= batSiz)
        assert max([len(t) for t in thmTxt]) <= batTim - 1  # reserve one place for [cls]
        vocabulary |= set("".join(thmTxt))

    # load similarity: predict similarity by given two sentences
    global simTrnSenA, simTrnSenB, simValSenA, simValSenB, simTrnCls, simValCls, simTrnSiz, simValSiz
    with open('text_similarity.txt',
              encoding="utf-8") as file:  # each line: senA + \t + senB + \t + similarity 0 / 1
        simSenA, simSenB, simCls = [], [], []
        for line in file:
            line = line.strip()  # remove trailing newline / spaces
            if not line:  # skip empty lines
                continue
            lines = line.split("\t")  # split by tab
            if len(lines) != 3:  # safety check
                continue
            senA, senB, cls = lines
            simSenA.append(senA)
            simSenB.append(senB)
            simCls.append(int(cls))

        simTrnSiz = int(len(simCls) * splRat)
        simValSiz = len(simCls) - simTrnSiz
        simTrnSenA, simValSenA = simSenA[:simTrnSiz], simSenA[simTrnSiz:]
        simTrnSenB, simValSenB = simSenB[:simTrnSiz], simSenB[simTrnSiz:]
        simTrnCls, simValCls = simCls[:simTrnSiz], simCls[simTrnSiz:]

        assert (len(simTrnCls) >= batSiz)
        assert (len(simValCls) >= batSiz)
        vocabulary |= set("".join(simSenA) + "".join(simSenB))

    # load answering: predict start and end on context by given context and question
    global ansTrnTxt, ansValTxt, ansTrnQue, ansValQue, ansTrnStr, ansValStr, ansTrnEnd, ansValEnd, ansTrnSiz, ansValSiz  # answer-training-context, ...
    with open('text_answering.txt',
              encoding="utf-8") as file:  # each line: [text ◟answer◞ text, \tab, question]
        ansTxt, ansQue, ansStr, ansEnd = [], [], [], []
        for line in file:
            line = line.strip()  # remove trailing newline / spaces
            if not line:  # skip empty lines
                continue
            lines = line.split("\t")  # split by tab
            if len(lines) != 2:  # safety check
                continue
            txt, que = lines

            start = txt.find('◟')
            end = txt.find('◞') - 1
            txt = txt.replace('◟', '').replace('◞', '')
            ansTxt.append(txt)
            ansQue.append(que)
            ansStr.append(start)
            ansEnd.append(end)

        ansTrnSiz = int(len(ansTxt) * splRat)
        ansTrnTxt = ansTxt[:ansTrnSiz]
        ansValTxt = ansTxt[ansTrnSiz:]
        ansTrnQue = ansQue[:ansTrnSiz]
        ansValQue = ansQue[ansTrnSiz:]
        ansTrnStr = ansStr[:ansTrnSiz]
        ansValStr = ansStr[ansTrnSiz:]
        ansTrnEnd = ansEnd[:ansTrnSiz]
        ansValEnd = ansEnd[ansTrnSiz:]
        ansTrnSiz = ansTrnSiz
        ansValSiz = len(ansValTxt)

        assert (len(ansTxt) >= batSiz)
        vocabulary |= set("".join(ansTxt) + "".join(ansQue))

    # load entity: [context] -> entity-marks for every token
    global entNam, entTrnTxt, entValTxt, entTrnMrk, entValMrk, entTrnSiz, entValSiz, numEnt, idx2ent, ent2idx  # entity-training-text, ...
    with open('text_entity.txt',
              encoding="utf-8") as file:  # first line: [name1, \t, name2, \t, ...]    other line pair: text + \n + marks

        # entity names
        entNam = file.readline().strip().split(', ')
        numEnt = len(entNam)  # number-of-entities
        idx2ent = {i: n for i, n in enumerate(entNam)}
        ent2idx = {n: i for i, n in idx2ent.items()}

        # text & entity-marks
        entTxt, entMrk = [], []
        while True:  # start from the second line
            txt = file.readline()
            mrk = file.readline()
            if (not txt) or (not mrk):  # EOF
                break
            txt, mrk = txt.strip(), mrk.strip()
            if (not txt) or (not mrk):  # skip blanks
                continue
            entTxt.append(txt)
            entMrk.append([int(m) for m in mrk])

        # split
        entTrnSiz = int(len(entTxt) * splRat)
        entTrnTxt = entTxt[:entTrnSiz]
        entValTxt = entTxt[entTrnSiz:]
        entTrnMrk = entMrk[:entTrnSiz]
        entValMrk = entMrk[entTrnSiz:]
        entTrnSiz = entTrnSiz
        entValSiz = len(entValTxt)

        assert (len(entTrnTxt) >= batSiz)
        assert (len(entValTxt) >= batSiz)
        vocabulary |= set("".join(entTxt))

    # load relation text
    global rltTrnTxt, rltValTxt, rltTrnCls, rltValCls, rltTrnTokMrk, rltValTokMrk, rltTrnSiz, rltValSiz, numRlt
    with open('text_relation.txt',
              encoding="utf-8") as file:  # first line: list of relations.   other line: text ◟entity1◞ text ⌞entity2⌟ text + \tab + relation-classification
        # relation names
        rltNam = file.readline().split(', ')
        numRlt = len(rltNam)
        idx2rlt = {i: n for i, n in enumerate(rltNam)}
        rlt2idx = {n: i for i, n in idx2rlt.items()}

        rltTxt, rltTokMrk, rltCls = [], [], []
        for line in file:
            line = line.strip()  # remove trailing newline / spaces
            if not line:  # skip empty lines
                continue
            lines = line.split("\t")  # split by tab
            if len(lines) != 2:  # safety check
                continue
            txt, cls = lines

            # text, token-mark
            str1 = txt.find('◟')
            end1 = txt.find('◞')
            str2 = txt.find('⌞')
            end2 = txt.find('⌟')
            txt = txt.replace('◟', '').replace('◞', '').replace('⌞', '').replace('⌟', '')
            mrk = [3] + [0] * (str1 - 1) + [1] * (end1 - str1) + [0] * (str2 - end1 - 1) + [2] * (end2 - str2)
            mrk += [0] * max(0, batTim - len(mrk))  # add paddings

            rltTokMrk.append(mrk)
            rltTxt.append(txt)
            rltCls.append(int(cls))

        assert len(rltTxt) == len(rltCls) == len(rltTokMrk)

        fullSize = min(len(rltTxt), len(rltCls), len(rltTokMrk))
        rltTrnSiz = int(fullSize * splRat)
        rltTrnTxt = rltTxt[:rltTrnSiz]
        rltValTxt = rltTxt[rltTrnSiz:]
        rltTrnCls = rltCls[:rltTrnSiz]
        rltValCls = rltCls[rltTrnSiz:]
        rltTrnTokMrk = rltTokMrk[:rltTrnSiz]
        rltValTokMrk = rltTokMrk[rltTrnSiz:]
        rltTrnSiz = len(rltTrnTxt)
        rltValSiz = len(rltValTxt)

        assert (len(rltCls) >= batSiz)
        assert (max([len(t) for t in rltTxt]) <= batTim - 1)

        vocabulary |= set("".join(rltTxt))

    # Tokenization
    global tokTbl, dtkTbl, tokenizer1, tokenizer2, detokenizer1, detokenizer2, vocSiz

    vocabulary = sorted(vocabulary - {'◟', '◞', '⌞', '⌟'})  # set -> list
    tokTbl = {'[pad]': 0,  # tokenization-table: padding
              '[unk]': 1,  # unknown
              '[cls]': 2,  # classification
              '[sep]': 3,  # separation
              '[msk]': 4,  # masking
              '◟': 5,  # entity-starting-1
              '◞': 6,  # entity-starting-1
              '⌞': 7,  # entity-ending-1
              '⌟': 8,  # entity-ending-2
              **{c: i + 9 for i, c in enumerate(vocabulary)}}  # {character : index + 9}
    vocSiz = len(tokTbl)

    tokenizer1 = lambda s: [tokTbl.get(c, tokTbl['[unk]']) for c in s]  # for single input string, like 'abcd'
    tokenizer2 = lambda b: [[tokTbl.get(c, tokTbl['[unk]']) for c in s] for s in
                            b]  # for bated input / list of strings, like ['abcd', 'efg', 'hijk', ...]

    dtkTbl = {i: c for c, i in tokTbl.items()}
    detokenizer1 = lambda t: ''.join([dtkTbl.get(i, '[unk]') for i in t])  # for single input list, like [1, 2, 3, ...]
    detokenizer2 = lambda b: [''.join([dtkTbl.get(i, '[unk]') for i in t]) for t in
                              b]  # for single input list, like [[1,2], [3,4,5], ...]

    preTrnTxt = tokenizer1(preTrnTxt)
    preValTxt = tokenizer1(preValTxt)

    thmTrnTxt = tokenizer2(thmTrnTxt)
    thmValTxt = tokenizer2(thmValTxt)

    simTrnSenA = tokenizer2(simTrnSenA)
    simTrnSenB = tokenizer2(simTrnSenB)
    simValSenA = tokenizer2(simValSenA)
    simValSenB = tokenizer2(simValSenB)

    ansTrnTxt = tokenizer2(ansTrnTxt)
    ansValTxt = tokenizer2(ansValTxt)
    ansTrnQue = tokenizer2(ansTrnQue)
    ansValQue = tokenizer2(ansValQue)

    entTrnTxt = tokenizer2(entTrnTxt)
    entValTxt = tokenizer2(entValTxt)

    rltTrnTxt = tokenizer2(rltTrnTxt)
    rltValTxt = tokenizer2(rltValTxt)

    # sentence separation for NSP
    global nspSen, nspSenLen, nspSenSiz
    txt = preTxt  # here, preTxt is text, not tokens.
    for sep in '，,、；;：:。．.？?！!\n¿¡‽⸮⸗⸚⸜⸝':
        txt = txt.replace(sep, sep + '⌂')
    for exc in ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.']:
        txt = txt.replace(exc + '⌂', exc)
    for exc in '’”':
        txt = txt.replace('⌂' + exc, exc + '⌂')
    txt = txt.replace('⌂\n⌂', '⌂')
    nspSen = [s.strip() for s in txt.split('⌂')]  # sentence
    newSen = []
    for s in nspSen:
        if len(s) <= (batTim - 3) / 2:  # reserve spaces for '[cls]', '[sep]', and '[sep]' in NSP
            newSen.append(s)
        else:
            start = 0
            while start < len(s):
                end = int(start + int((batTim - 3) / 2))
                newSen.append(s[start:end])
                start = end
    nspSen = tokenizer2(newSen)  # nsp-sentences
    nspSenLen = [len(s) for s in nspSen]  # list of all sentence lengths
    nspSenSiz = len(nspSen)
    if not os.path.exists('_nsp_sentences.txt'):
        with open('_nsp_sentences.txt', 'w', encoding='utf-8') as file:  # export the sentences for check
            for s in nspSen:
                file.write('\n\n==========\n\n' + detokenizer1(s))

    # sentence split for NSP
    global nspTrnSen, nspValSen, nspTrnSenLen, nspValSenLen, nspTrnSiz, nspValSiz
    simTrnSiz = int(nspSenSiz * splRat)  # split-size for nsp sentences
    nspTrnSen = nspSen[:simTrnSiz]  # nsp-training-sentences
    nspValSen = nspSen[simTrnSiz:]  # validation-sentences
    nspTrnSenLen = nspSenLen[:simTrnSiz]
    nspValSenLen = nspSenLen[simTrnSiz:]
    nspTrnSiz = simTrnSiz  # nsp-training-size: number of training sentences
    nspValSiz = len(nspValSen)


def batch_MLM(split):  # bat-for-masked-language-modeling: tokens masked with "[msk]" or other tokens,
    if split == 'trn':
        txt, rng = preTrnTxt, preTrnTxtLen - (batTim - 1)
    else:
        txt, rng = preValTxt, preValTxtLen - (batTim - 1)

    start = torch.randint(0, max(1, rng), (batSiz,))  # starting-timepoints (batSiz,)
    start = start.tolist()

    orgTok = [[tokTbl['[cls]']] + txt[s: s + batTim - 1] for s in start]
    orgTok = torch.tensor(orgTok, dtype=torch.long)

    mskTok = orgTok.clone()  # masked-tokens
    tokFcs = torch.zeros_like(mskTok, dtype=torch.bool)  # token-focus for masked tokens: 0 to ignore, 1 to focus
    for i in range(batSiz):  # for each row of original-tokens
        for j in range(1, batTim):  # for each masked element
            if random.random() < mskRat:
                tokFcs[i][j] = 1  # focus on this token in cross_entropy()
                r = random.random()
                if r < 0.8:  # mask the token
                    mskTok[i][j] = tokTbl['[msk]']  # mask by '[msk]'
                elif r < 0.9:
                    mskTok[i][j] = random.randint(9, vocSiz - 1)  # mask by other tokens

    tokMrk = torch.cat([
        torch.full((batSiz, 1), 3, dtype=torch.long),  # [cls] token mark = 3
        torch.ones((batSiz, batTim - 1), dtype=torch.long)  # normal tokens = 1
    ], dim=1)

    return (mskTok.to(device),  # (batSiz, batTim)
            orgTok.to(device),  # (batSiz, batTim)
            tokFcs.to(device),  # (batSiz, batTim)
            tokMrk.to(device))  # (batSiz, batTim)


def batch_NSP(split):  # bat-for-next-sentence-prediction: add nspCls "[cls]", separation "⌂", and padding "[pad]"
    if split == 'trn':
        snt, numSnt, sntLen = nspTrnSen, nspTrnSiz, nspTrnSenLen
    else:
        snt, numSnt, sntLen = nspValSen, nspValSiz, nspValSenLen

    assert numSnt >= 2

    twoSen, senMrk, nspCls = [], [], []  # two-sentences, sentence-marks (1 or 2 for sentences / 0 for paddings / 3 for [cls] and ⌂) for each token, next-sentence-prediction-marks (0 for is-next, 1 for not-next)
    for _ in range(batSiz):  # for each row
        # choose sentence indices
        r1 = random.randint(0, numSnt - 2)  # index of the first sentence
        r2 = r1 + 1  # index of the second sentence

        # next-sentence-prediction-marks
        if random.random() < 0.5:
            nspCls.append(0)  # is next
        else:
            nspCls.append(1)  # not next
            if random.random() < 0.5:  # exchange r1 and r2
                r1, r2 = r2, r1
            else:  # change r2
                while r1 + 1 == r2 or r1 == r2:
                    r2 = random.randint(0, numSnt - 1)

        # twoSen
        tokens = [tokTbl['[cls]'], *snt[r1], tokTbl['[sep]'], *snt[r2], tokTbl['[sep]']]
        if len(tokens) > batTim:
            tokens = tokens[:batTim]
        numPad = batTim - len(tokens)  # number-of-paddings
        if numPad > 0:  # if padding exists
            tokens += [tokTbl['[pad]']] * numPad  # add paddings
        twoSen.append(tokens)

        # token-marks
        tokIds = []
        senFlg = 1  # sentence-index
        for t in tokens:
            if t in (tokTbl['[cls]'], tokTbl['[sep]']):  # for special tokens
                tokIds.append(3)
            elif t == tokTbl['[pad]']:  # for paddings
                tokIds.append(0)
            else:  # for sentence tokens
                tokIds.append(senFlg)

            if t == tokTbl['[sep]'] and senFlg == 1:
                senFlg = 2

        senMrk.append(tokIds)

    return (torch.tensor(twoSen, dtype=torch.long, device=device),  # (batSiz, batTim)
            torch.tensor(senMrk, dtype=torch.long, device=device),  # (batSiz, batTim)
            torch.tensor(nspCls, dtype=torch.long, device=device))  # (batSiz,)


def batch_theme(split):
    if split == 'trn':
        text, cls, size = thmTrnTxt, thmTrnCls, thmTrnSiz
    else:
        text, cls, size = thmValTxt, thmValCls, thmValSiz

    thmTok, tokMrk, themes = [], [], []

    ids = random.sample(range(size), batSiz)  # non-repeated indices
    for i in ids:  # for each line
        toks = [tokTbl['[cls]']] + text[i][:batTim - 2] + [tokTbl['[sep]']]
        numPad = max(0, batTim - len(toks))
        toks += [tokTbl['[pad]']] * numPad  # add paddings
        thmTok.append(toks)

        marks = []
        for t in toks:
            if t == tokTbl['[cls]']:
                marks.append(3)
            elif t == tokTbl['[pad]']:
                marks.append(0)
            else:
                marks.append(1)
        tokMrk.append(marks)

        themes.append(cls[i])

    return (torch.tensor(thmTok, dtype=torch.long, device=device),
            torch.tensor(tokMrk, dtype=torch.long, device=device),
            torch.tensor(themes, dtype=torch.long, device=device))


def batch_similarity(split):
    if split == 'trn':
        senA, senB, cls, numSnt = simTrnSenA, simTrnSenB, simTrnCls, simTrnSiz
    else:
        senA, senB, cls, numSnt = simValSenA, simValSenB, simValCls, simValSiz

    simSenA, simSenMrkA, simSenB, simSenMrkB, simCls = [], [], [], [], []

    rows = random.sample(range(len(senA)), min(batSiz, len(senA)))
    for r in rows:
        tokA = [tokTbl['[cls]']] + senA[r][:batTim - 2] + [tokTbl['[sep]']]
        tokB = [tokTbl['[cls]']] + senB[r][:batTim - 2] + [tokTbl['[sep]']]

        tokMrkA = [3] + [1] * (len(tokA) - 2) + [3]
        tokMrkB = [3] + [1] * (len(tokB) - 2) + [3]

        numPadA = max(0, batTim - len(tokA))  # avoid negative padding
        numPadB = max(0, batTim - len(tokB))

        tokA = tokA[:batTim] + [tokTbl['[pad]']] * numPadA  # truncate + pad
        tokB = tokB[:batTim] + [tokTbl['[pad]']] * numPadB

        tokMrkA = tokMrkA[:batTim] + [0] * numPadA
        tokMrkB = tokMrkB[:batTim] + [0] * numPadB

        # appending
        simSenA.append(tokA)
        simSenB.append(tokB)
        simSenMrkA.append(tokMrkA)
        simSenMrkB.append(tokMrkB)
        simCls.append(cls[r])  # Dummy similarity score (replace with real simCls)

    return (torch.tensor(simSenA, device=device),
            torch.tensor(simSenMrkA, device=device),
            torch.tensor(simSenB, device=device),
            torch.tensor(simSenMrkB, device=device),
            torch.tensor(simCls, device=device))


def batch_QA(split):  # question-answering batch
    if split == 'trn':  # choose training split
        texts, questions, start, end, size = ansTrnTxt, ansTrnQue, ansTrnStr, ansTrnEnd, ansTrnSiz
    else:  # validation split
        texts, questions, start, end, size = ansValTxt, ansValQue, ansValStr, ansValEnd, ansValSiz

    batCtx, batCtxMrk, batStr, batEnd = [], [], [], []  # batch-for-context, ...

    rows = random.sample(range(size), batSiz)
    for r in rows:
        ctx = ([tokTbl['[cls]']] + texts[r] + [tokTbl['[sep]']] + questions[r] + [tokTbl['[sep]']])[:batTim]  # context
        numPad = batTim - len(ctx)
        ctx += [tokTbl['[pad]']] * numPad
        batCtx.append(ctx)

        ctxMrk = []  # context-marks
        for t in ctx:
            if t == tokTbl['[cls]'] or t == tokTbl['[sep]']:
                ctxMrk.append(3)
            elif t == tokTbl['[pad]']:
                ctxMrk.append(0)
            else:
                ctxMrk.append(1)
        batCtxMrk.append(ctxMrk)

        batStr.append(start[r] + 1)  # because of newly added '[cls]' token
        batEnd.append(end[r] + 1)

    return (torch.tensor(batCtx, device=device),
            torch.tensor(batCtxMrk, device=device),
            torch.tensor(batStr, device=device),
            torch.tensor(batEnd, device=device))


def batch_NER(split):  # named entity recognition
    if split == 'trn':
        text, entMrk, size = entTrnTxt, entTrnMrk, entTrnSiz
    else:
        text, entMrk, size = entValTxt, entValMrk, entValSiz

    batTxt, tokMrk, entMrks = [], [], []

    rows = random.sample(range(size), batSiz)
    for r in rows:
        toks = [tokTbl['[cls]']] + text[r][:batTim - 2] + [tokTbl['[sep]']]
        tMrk = [3] + [1] * (len(toks) - 2) + [3]  # token-marks
        eMrk = [0] + entMrk[r][:len(toks) - 2] + [0]  # entity-marks

        # padding
        numPad = batTim - len(toks)
        toks += [tokTbl['[pad]']] * numPad
        tMrk += [0] * numPad
        eMrk += [0] * numPad

        batTxt.append(toks)
        tokMrk.append(tMrk)
        entMrks.append(eMrk)

    return (torch.tensor(batTxt, device=device),
            torch.tensor(tokMrk, device=device),  # token-marks for separation embedding
            torch.tensor(entMrks, device=device))  # entity-marks are targets for NER prediction


def batch_relation(split):  # relation extraction：context + start-end -> name classification
    if split == 'trn':  # choose training split
        txt, cls, tokMrk, size = rltTrnTxt, rltTrnCls, rltTrnTokMrk, rltTrnSiz
    else:  # choose validation split
        txt, cls, tokMrk, size = rltValTxt, rltValCls, rltValTokMrk, rltValSiz

    batTxt, batCls, batTokMrk = [], [], []

    ids = random.sample(range(size), min(batSiz, size))
    for i in ids:
        toks = [tokTbl['[cls]']] + txt[i][:batTim - 2] + [tokTbl['[sep]']]
        numPad = max(0, batTim - len(toks))
        toks += [tokTbl['[pad]']] * numPad

        batTxt.append(toks)
        batCls.append(cls[i])
        batTokMrk.append(tokMrk[i][:batTim])

    return (torch.tensor(batTxt, device=device),
            torch.tensor(batTokMrk, device=device),
            torch.tensor(batCls, device=device))


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokEmbTbl = nn.Embedding(len(tokTbl), embDim,
                                      padding_idx=0)  # token-embedding-table: [pad] -> [0, 0, ... , 0]
        self.timEmbTbl = nn.Embedding(batTim, embDim)  # time-embedding-table (also called position-embedding)
        self.senEmbTbl = nn.Embedding(3, embDim,
                                      padding_idx=0)  # sentence-embedding-table:  0: paddings and special tokens.  1: sentence A.  2: sentence B.
        self.lyrNrm = nn.LayerNorm(embDim)
        self.drp = nn.Dropout(drpRat)

    def forward(self, batTok, tokMrk):  # bated-tokens (batSiz, batTim), sentence-indices (batSiz, batTim)
        tokEmb = self.tokEmbTbl(batTok)  # (batSiz,batTim,embDim)
        device = batTok.device
        timIds = torch.arange(batTok.size(1), device=device)  # time-indices (batTim,)
        timEmb = self.timEmbTbl(timIds).unsqueeze(0)  # time-embeddings (1, batTim, embDim)
        padMsk = (batTok != 0).unsqueeze(-1)  # (batSiz, batTim, 1)
        padMsk = padMsk.to(device=device, dtype=tokEmb.dtype)
        batEmb = tokEmb + timEmb * padMsk  # zero out paddings
        senEmb = self.senEmbTbl(tokMrk.masked_fill(tokMrk == 3, 0))  # (batSiz,batTim,embDim)
        batEmb = batEmb + senEmb  # (batSiz,batTim,embDim)
        batEmb = self.lyrNrm(batEmb)
        batEmb = self.drp(batEmb)
        return batEmb


class Attention_Head(nn.Module):
    def __init__(self):
        super().__init__()

        # parameters
        self.query = nn.Linear(embDim, heaSiz, bias=False)
        self.key = nn.Linear(embDim, heaSiz, bias=False)
        self.value = nn.Linear(embDim, heaSiz, bias=False)
        self.dropout = nn.Dropout(drpRat)

        # lora
        self.lraQue = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, heaSiz, bias=False))
        self.lraKey = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, heaSiz, bias=False))
        self.lraVal = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, heaSiz, bias=False))
        self.lraDrp = nn.Dropout(lraDrp)

        for seq in [self.lraQue, self.lraKey, self.lraVal]:
            nn.init.kaiming_uniform_(seq[0].weight, a=math.sqrt(5))
            nn.init.zeros_(seq[1].weight)

    def forward(self, batEmb, tokMrk):
        q = self.query(batEmb) + self.lraQue(self.lraDrp(batEmb)) * lraScl * lora  # (batSiz, batTim, heaSiz)
        k = self.key(batEmb) + self.lraKey(self.lraDrp(batEmb)) * lraScl * lora  # (batSiz, batTim, heaSiz)
        v = self.value(batEmb) + self.lraVal(self.lraDrp(batEmb)) * lraScl * lora  # (batSiz, batTim, heaSiz)

        w = (q @ k.transpose(-2, -1)) * (heaSiz ** -0.5)  # (batSiz, batTim, batTim)
        attMsk = tokMrk == 0  # attention-mask (batSiz, batTim), True for padding token
        w = w.masked_fill(attMsk.unsqueeze(1), float('-inf'))  # mask padding tokens
        w = F.softmax(w, dim=-1)  # softmax over last dimension
        w = self.dropout(w)
        w = torch.nan_to_num(w, nan=0.0)  # remove NaNs if any

        return w @ v


class Attention_Multi_Head(nn.Module):  # multi-head attention
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention_Head() for _ in range(numHea)])
        self.proj = nn.Linear(numHea * heaSiz, embDim)
        self.lraPrj = nn.Sequential(nn.Linear(numHea * heaSiz, lraRnk, bias=False),
                                    nn.Linear(lraRnk, embDim, bias=False))  # lora-projection
        nn.init.zeros_(self.lraPrj[1].weight)
        self.lraDrp = nn.Dropout(lraDrp)

    def forward(self, batEmb, tokMrk):
        batAdj = torch.cat([h(batEmb, tokMrk) for h in self.heads], dim=-1)
        return self.proj(batAdj) + self.lraPrj(self.lraDrp(batAdj)) * lraScl * lora


class FFN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.lyr1 = nn.Linear(embDim, 4 * embDim)
        self.activation = nn.GELU()
        self.lyr2 = nn.Linear(4 * embDim, embDim)
        self.dropout = nn.Dropout(drpRat)

        # lora
        self.lraLyr1 = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False),
                                     nn.Linear(lraRnk, 4 * embDim, bias=False))  # lora-projection
        self.lraLyr2 = nn.Sequential(nn.Linear(4 * embDim, lraRnk, bias=False),
                                     nn.Linear(lraRnk, embDim, bias=False))  # lora-projection
        nn.init.zeros_(self.lraLyr1[1].weight)
        nn.init.zeros_(self.lraLyr2[1].weight)
        self.lraDrp1 = nn.Dropout(lraDrp)
        self.lraDrp2 = nn.Dropout(lraDrp)

    def forward(self, batEmb):
        batEmb = self.lyr1(batEmb) + self.lraLyr1(self.lraDrp1(batEmb)) * lraScl * lora
        batEmb = self.activation(batEmb)
        batEmb = self.lyr2(batEmb) + self.lraLyr2(self.lraDrp2(batEmb)) * lraScl * lora
        batEmb = self.dropout(batEmb)
        return batEmb


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = Attention_Multi_Head()
        self.ffn = FFN()
        self.ln1 = nn.LayerNorm(embDim)
        self.ln2 = nn.LayerNorm(embDim)

    def forward(self, batEmb, tokMrk):
        batEmb = batEmb + self.mh(self.ln1(batEmb), tokMrk)  # residual connection
        batEmb = batEmb + self.ffn(self.ln2(batEmb))  # residual connection
        return batEmb


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embeddings()
        self.layers = nn.ModuleList([Block() for _ in range(numLyr)])

    def forward(self, batTok, tokMrk):
        batEmb = self.embed(batTok, tokMrk)
        for layer in self.layers:
            batEmb = layer(batEmb, tokMrk)
        return batEmb  # bate-embeddings after adjustments


class MLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm_head = nn.Sequential(nn.Linear(embDim, embDim), nn.GELU(), nn.LayerNorm(embDim),
                                      nn.Linear(embDim, vocSiz))

    def forward(self, spl):
        mskTok, orgTok, tokFcs, tokMrk = batch_MLM(spl)
        adjEmb = transformer(mskTok, tokMrk)
        logits = self.mlm_head(adjEmb)  # (batSiz, batTim, vocSiz)
        logits = logits.view(-1, vocSiz)  # (batSiz * batTim, vocSiz)
        tokFcs = tokFcs.view(-1)
        orgTok = orgTok.view(-1)  # (batSiz * batTim,)
        loss = F.cross_entropy(logits[tokFcs], orgTok[tokFcs])
        return loss


class NSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(embDim, embDim)
        self.activation = nn.Tanh()  # elements bounded to [-1, 1]
        self.head = nn.Linear(embDim, 2)

    def forward(self, spl):  # adjusted-embeddings (batSiz, batTim, embDim), bated-classification-NSP (batSiz,)
        tok, tokMrk, cls = batch_NSP(spl)
        adjEmb = transformer(tok, tokMrk)
        logits = self.head(self.activation(self.dense(adjEmb[:, 0])))  # (batSiz, 2)
        loss = F.cross_entropy(logits, cls)
        return loss


class Theme_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(embDim, embDim)
        self.act = nn.Tanh()
        self.classifier = nn.Linear(embDim, numThm)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, spl, mode='loss'):
        tok, tokMrk, cls = batch_theme(spl)
        enc = transformer(tok, tokMrk)  # (batSiz, batTim, embDim)
        tokCls = enc[:, 0]  # token of [cls]
        logits = self.classifier(self.act(self.dense(tokCls)))  # (batSiz, numCls)

        if mode == 'loss':
            return self.loss(logits, cls)
        else:
            prdCls = logits.argmax(dim=1)  # predicted-classification
            accuracy = (prdCls == cls).float().mean().item()
            print(f'theme accuracy: {accuracy}')


class Sentence_Similarity(nn.Module):
    def __init__(self):
        super().__init__()

        self.prj = nn.Linear(embDim, embDim, bias=True)
        self.act = nn.Tanh()  # activation: normalization

        self.lraA = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, embDim, bias=False))  # lora
        nn.init.kaiming_uniform_(self.lraA[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lraA[1].weight)

        self.lraB = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, embDim, bias=False))  # lora
        nn.init.kaiming_uniform_(self.lraB[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lraB[1].weight)

        self.lraDrp = nn.Dropout(lraDrp)
        self.loss = nn.MSELoss()  # mean-squared-error

    def forward(self, spl, mode='loss'):  # sentence-A, sentence-marks-A, sentence-B, sentence-marks-B, all are of shape (batSiz, batTim, embDim)
        senA, senMrkA, senB, senMrkB, cls = batch_similarity(spl)

        encA = transformer(senA, senMrkA)  # encoded-sentence-A (bat, batTim, embDim)
        encB = transformer(senB, senMrkB)  # (bat, batTim, embDim)

        tokClsA = encA[:, 0]  # (batSiz, embDim), sentence pooling: let the classification token '[cls]' represent each sentence.
        tokClsB = encB[:, 0]  # (batSiz, embDim)

        prjA = self.act(self.prj(tokClsA) + self.lraA(self.lraDrp(tokClsA)) * lraScl * lora)  # (batSiz, embDim)
        prjB = self.act(self.prj(tokClsB) + self.lraB(self.lraDrp(tokClsB)) * lraScl * lora)  # (batSiz, embDim)

        logits = F.cosine_similarity(prjA, prjB, dim=-1)  # (batSiz,)  semantic similarity scores [-1,1]

        if mode == 'loss':
            trgCls = cls.float() * 2 - 1  # interval [0, 1] -> interval [-1, 1] for cosine similarity
            return self.loss(logits, trgCls)  # MSE loss for semantic similarity
        else:
            prdCls = (logits >= 0)
            accuracy = (prdCls == cls).mean().item()
            print(f'similarity accuracy: {accuracy}')


class Question_Answering(nn.Module):
    def __init__(self):
        super().__init__()
        self.qa_head = nn.Linear(embDim, 2)  # predicts [start, end]
        self.lra_head = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, 2, bias=False))
        nn.init.kaiming_uniform_(self.lra_head[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lra_head[1].weight)
        self.lraDrp = nn.Dropout(lraDrp)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, spl, mode='loss'):
        ctx, ctxMrk, start, end = batch_QA(spl)
        encCtx = transformer(ctx, ctxMrk)  # encodings (batSiz, batTim, embDim)
        logits = self.qa_head(encCtx) + self.lra_head(self.lraDrp(encCtx)) * lraScl * lora
        lgtStr, lgtEnd = logits.split(1,
                                      dim=-1)  # logits-of-startings (batSiz, batTim, 1), logits-of-endings (batSiz, batTim, 1)
        lgtStr = lgtStr.squeeze(-1)  # (batSiz, batTim)
        lgtEnd = lgtEnd.squeeze(-1)  # (batSiz, batTim)
        if mode == 'loss':
            return (self.loss(lgtStr, start) + self.loss(lgtEnd, end)) / 2
        else:
            print(f'Q.A. accuracy:',
                  ((lgtStr.argmax(dim=1) == start) & (lgtEnd.argmax(dim=1) == end)).float().mean().item())


class Named_Entity_Recognition(nn.Module):
    def __init__(self):  # number-of-labels
        super().__init__()
        self.classifier = nn.Linear(embDim, numEnt)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, spl, mode='loss'):
        tok, tokMrk, entMrk = batch_NER(spl)
        enc = transformer(tok, tokMrk)  # encodings (batSiz, batTim, embDim)
        logits = self.classifier(enc)  # (batSiz, batTim, numLbl)
        if mode == 'loss':
            return self.loss(logits.view(-1, logits.size(-1)),
                             entMrk.view(-1))  # (batSiz * batTim, numLbl), (batSiz * batTim)
        else:
            mask = tokMrk != 0  # ignore [pad]
            acc = ((logits.argmax(dim=-1) == entMrk) * mask).float().sum() / mask.float().sum()
            print('entity accuracy (no pad):', acc.item())


class Relation_Prediction(nn.Module):
    def __init__(self):  # number-of-labels
        super().__init__()
        self.dense = nn.Linear(embDim, embDim)
        self.lra_dense = nn.Sequential(nn.Linear(embDim, lraRnk, bias=False), nn.Linear(lraRnk, embDim, bias=False))
        nn.init.kaiming_uniform_(self.lra_dense[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lra_dense[1].weight)
        self.lraDrp = nn.Dropout(lraDrp)
        self.act = nn.Tanh()
        self.classifier = nn.Linear(embDim, numRlt)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, spl, mode='loss'):
        tok, tokMrk, cls = batch_relation(spl)
        enc = transformer(tok, tokMrk)  # (batSiz, batTim, embDim)
        sen = enc[:, 0]  # sentence-head [cls] (batSiz, embDim)
        prj = self.act(self.dense(sen) + self.lra_dense(self.lraDrp(sen)) * lraScl * lora)
        logits = self.classifier(prj)

        if mode == 'loss':
            return self.loss(logits, cls)
        else:
            acc = (logits.argmax(dim=1) == cls).float().mean()
            print('relation accuracy:', acc.item())


def models():
    # create models
    global transformer, mlm, nsp, theme, similarity, answering, entity, relation
    transformer = Transformer().to(device)
    mlm = MLM().to(device)
    nsp = NSP().to(device)
    theme = Theme_Classification().to(device)
    similarity = Sentence_Similarity().to(device)
    answering = Question_Answering().to(device)
    entity = Named_Entity_Recognition().to(device)
    relation = Relation_Prediction().to(device)

    # load parameters
    if os.path.isfile('_parameters'):
        parameters = torch.load('_parameters', map_location=device)
        print("parameter checkpoint loaded\n")
        transformer.load_state_dict(parameters['transformer'])
        mlm.load_state_dict(parameters['mlm'])
        nsp.load_state_dict(parameters['nsp'])
        theme.load_state_dict(parameters['theme'])
        similarity.load_state_dict(parameters['similarity'])
        answering.load_state_dict(parameters['answering'])
        entity.load_state_dict(parameters['entity'])
        relation.load_state_dict(parameters['relation'])
    else:
        print("no existing parameter checkpoint\n")


def lora_switch(l):  #
    global lora  # 0 to turn off lora, and 1 to turn on
    lora = float(l)
    for model in [transformer, mlm, nsp, theme, similarity, answering, entity, relation]:
        for name, param in model.named_parameters():
            if "lra" in name:  # LoRA parameters
                param.requires_grad = bool(l)
            else:  # pretrained transformer params
                param.requires_grad = not bool(l)


def lssEst_pretrain():  # loss-estimation for pretrain
    transformer.eval();
    mlm.eval();
    nsp.eval()

    lssTrn, lssVal = [], []
    with torch.no_grad():
        for _ in range(itrLss[0]):
            for spl in ['trn', 'val']:
                loss = (mlm(spl) + nsp(spl)).item()
                if spl == 'trn':
                    lssTrn.append(loss)
                else:
                    lssVal.append(loss)

    transformer.train();
    mlm.train();
    nsp.train()
    return [sum(lssTrn) / len(lssTrn), sum(lssVal) / len(lssVal)]


def lssEst_finetune():  # loss-estimation for pretrain
    transformer.eval();
    theme.eval();
    similarity.eval();
    answering.eval();
    entity.eval();
    relation.eval()

    lssTrn, lssVal = [], []
    with torch.no_grad():
        for _ in range(itrLss[1]):
            for spl in ['trn', 'val']:
                loss = (theme(spl, 'loss') + similarity(spl, 'loss') + answering(spl, 'loss') + entity(spl,
                                                                                                       'loss') + relation(
                    spl, 'loss')).item()
                if spl == 'trn':
                    lssTrn.append(loss)
                else:
                    lssVal.append(loss)

    transformer.train();
    theme.train();
    similarity.train();
    answering.train();
    entity.train();
    relation.train()
    return [sum(lssTrn) / len(lssTrn), sum(lssVal) / len(lssVal)]


def pretrain():
    lora_switch(0)
    optimizer = torch.optim.AdamW(itertools.chain(transformer.parameters(), mlm.parameters(), nsp.parameters()),
                                  lr=stpLen[0])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / 100))
    for i in range(itrOpt[0]):  # adjust epochs
        transformer.train();
        mlm.train();
        nsp.train()
        loss = mlm('trn') + nsp('trn')
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*transformer.parameters(), *mlm.parameters(), *nsp.parameters()], 1.0)
        optimizer.step()
        scheduler.step()

        # loss observation
        if (i + 1) % obsOpt[0] == 0:  # parameters observation
            lssObs = lssEst_pretrain()  # loss-observation
            print(
                f"pretrain: step {i + 1}: train bertPretrain {lssObs[0]:.4f}, validation bertPretrain {lssObs[1]:.4f}")


def full_fine_tune():
    lora_switch(0)  # turn off lora
    optimizer = torch.optim.AdamW(
        itertools.chain(theme.parameters(), similarity.parameters(), answering.parameters(), entity.parameters(),
                        relation.parameters()), lr=stpLen[1])
    for i in range(itrOpt[1]):
        transformer.train();
        theme.train();
        similarity.train();
        answering.train();
        entity.train();
        relation.train()  # training mode
        loss = theme('trn', 'loss') + similarity('trn', 'loss') + answering('trn', 'loss') + entity('trn',
                                                                                                    'loss') + relation(
            'trn', 'loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(itertools.chain(transformer.parameters(), theme.parameters(), similarity.parameters(),
                                                 answering.parameters(), entity.parameters(), relation.parameters()),
                                 1.0)  # clip gradient norm less than 1.0
        optimizer.step()

        # loss observation
        if (i + 1) % obsOpt[1] == 0:  # parameters observation
            lssObs = lssEst_finetune()  # loss-observation
            print(
                f"full fine tune: step {i + 1}: train bertPretrain {lssObs[0]:.4f}, validation bertPretrain {lssObs[1]:.4f}")


def lora_fine_tune():
    lora_switch(1)  # freeze transformer, only LoRA trains
    optimizer = torch.optim.AdamW(
        itertools.chain(theme.parameters(), similarity.parameters(), answering.parameters(), entity.parameters(),
                        relation.parameters()), lr=stpLen[2])
    for i in range(itrOpt[2]):  # adjust epochs
        transformer.train();
        theme.train();
        similarity.train();
        answering.train();
        entity.train();
        relation.train()  # training mode
        loss = theme('trn', 'loss') + similarity('trn', 'loss') + answering('trn', 'loss') + entity('trn',
                                                                                                    'loss') + relation(
            'trn', 'loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(itertools.chain(transformer.parameters(), theme.parameters(), similarity.parameters(),
                                                 answering.parameters(), entity.parameters(), relation.parameters()),
                                 1.0)  # clip gradient norm less than 1.0
        optimizer.step()

        # loss observation
        if (i + 1) % obsOpt[2] == 0:  # parameters observation
            lssObs = lssEst_finetune()  # loss-observation
            print(
                f"lora fine tune: step {i + 1}: train bertPretrain {lssObs[0]:.4f}, validation bertPretrain {lssObs[1]:.4f}")


def checkpoint():
    torch.save({'transformer': transformer.state_dict(),
                'mlm': mlm.state_dict(),
                'nsp': nsp.state_dict(),
                'theme': theme.state_dict(),
                'similarity': similarity.state_dict(),
                'answering': answering.state_dict(),
                'entity': entity.state_dict(),
                'relation': relation.state_dict(),
                }, '_parameters')
    print(
        "parameters checkpoint saved.")


hyp()
setup()
models()
pretrain()
full_fine_tune()
lora_fine_tune()
checkpoint()