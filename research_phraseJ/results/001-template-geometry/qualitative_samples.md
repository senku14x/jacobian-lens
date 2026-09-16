# 001 — qualitative samples

Read position = final prompt token. J-lens top tokens use the released J at the stated layer. Template rows = top cosine matches in the 13,731-row released vocabulary. Probe = whitened mean-difference direction fit on ALL latent prompts of the pair (in-sample, illustrative only; the report's numbers are held-out).


## New Zealand vs New York (latent, L56)

**New Zealand** — route `continent`  
> Geography note: the city of Auckland can be found on the continent called  
> model's next token: ` Oce` (expected `Oceania`)  
> probe score (a−b direction): +1359.63  |  J-sum New Zealand: 21.32 vs New York: 8.04  
> J-lens top-8 @L56: [' Zealand', '...', ' Auckland', '.nz', 'NZ', ' NZ', 'nz', ' nz']  
> template top-6 @L56: [('New Zealand', 0.3), ('zealand', 0.261), ('asia', 0.17), ('continent', 0.167), ('africa', 0.159), ('fiji', 0.148)]

**New Zealand** — route `hemisphere`  
> Fact: The hemisphere in which the place where the city of Auckland is located lies is the  
> model's next token: ` Southern` (expected `Southern`)  
> probe score (a−b direction): +1034.95  |  J-sum New Zealand: 14.51 vs New York: 3.71  
> J-lens top-8 @L56: [' hemisphere', ' Hemisphere', '半球', ' Northern', ' North', ' northern', '北', ' hem']  
> template top-6 @L56: [('hemisphere', 0.263), ('New Zealand', 0.197), ('zealand', 0.144), ('pacific', 0.138), ('americas', 0.104), ('auckland', 0.092)]

**New Zealand** — route `second_word_letters`  
> Fact: The number of letters in the second word of the name of the place where the Maori people is located is  
> model's next token: ` ` (expected `7`)  
> probe score (a−b direction): -1.73  |  J-sum New Zealand: -1.77 vs New York: -1.82  
> J-lens top-8 @L56: [' odd', ' divisible', '偶', ' greater', '奇', ' equal', 'odd', ' Odd']  
> template top-6 @L56: [('eleven', 0.162), ('forty', 0.14), ('twelve', 0.128), ('odd', 0.117), ('nine', 0.107), ('nineteen', 0.1)]

**New York** — route `continent`  
> Q: On which continent is the place known for Wall Street? A:  
> model's next token: `

` (expected `North America`)  
> probe score (a−b direction): -252.40  |  J-sum New Zealand: 4.51 vs New York: 5.23  
> J-lens top-8 @L56: ['<think>', '北美', ' continents', ' Europe', '美洲', ' Americas', '洲', ' continent']  
> template top-6 @L56: [('asia', 0.157), ('continent', 0.107), ('africa', 0.107), ('european', 0.085), ('europe', 0.084), ('asian', 0.07)]

**New York** — route `currency`  
> Fact: The currency used in the place where Wall Street is located is the  
> model's next token: ` US` (expected `dollar`)  
> probe score (a−b direction): -215.30  |  J-sum New Zealand: 3.26 vs New York: 3.67  
> J-lens top-8 @L56: [' dollar', ' USD', ' US', ' United', ' dollars', '美元', ' currency', 'USD']  
> template top-6 @L56: [('euro', 0.265), ('yuan', 0.238), ('sterling', 0.213), ('yen', 0.199), ('dollar', 0.196), ('usd', 0.171)]

**New York** — route `continent`  
> Geography note: the Empire State Building can be found on the continent called  
> model's next token: ` North` (expected `North America`)  
> probe score (a−b direction): -176.54  |  J-sum New Zealand: 6.84 vs New York: 5.18  
> J-lens top-8 @L56: ['...', '北美', ' Americas', '…', ' America', ' __', '美洲', '洲']  
> template top-6 @L56: [('asia', 0.206), ('continent', 0.189), ('africa', 0.184), ('european', 0.127), ('arabia', 0.105), ('europe', 0.091)]


## South Korea vs South Africa (latent, L60)

**South Korea** — route `second_word_letters`  
> Fact: The number of letters in the second word of the name of the place where Gyeongbokgung Palace is located is  
> model's next token: ` ` (expected `5`)  
> probe score (a−b direction): +116.25  |  J-sum South Korea: 2.17 vs South Africa: 0.24  
> J-lens top-8 @L60: [' greater', ' divisible', ' odd', ' equal', '偶', ' less', ' larger', 'odd']  
> template top-6 @L60: [('eleven', 0.147), ('forty', 0.116), ('twelve', 0.109), ('nine', 0.097), ('odd', 0.095), ('nineteen', 0.093)]

**South Korea** — route `hemisphere`  
> Geography note: the Samsung company is located in the  
> model's next token: ` city` (expected `Northern`)  
> probe score (a−b direction): +1474.13  |  J-sum South Korea: 28.19 vs South Africa: 18.59  
> J-lens top-8 @L60: [' South', ' south', ' Republic', ' southern', 'South', ' city', ' Gang', 'south']  
> template top-6 @L60: [('seoul', 0.241), ('korea', 0.168), ('korean', 0.106), ('netherlands', 0.094), ('philippine', 0.088), ('northwestern', 0.084)]

**South Korea** — route `second_word_letters`  
> Puzzle: the place where K-pop music is located has a two-word name; its second word has this many letters:  
> model's next token: ` ` (expected `5`)  
> probe score (a−b direction): +99.12  |  J-sum South Korea: 0.02 vs South Africa: -1.22  
> J-lens top-8 @L60: [' _', 'Hint', ' hint', '____', '._', ' clue', ' __', ' `_']  
> template top-6 @L60: [('eleven', 0.122), ('viii', 0.113), ('eight', 0.083), ('seven', 0.08), ('nine', 0.079), ('vii', 0.078)]

**South Africa** — route `hemisphere`  
> Fact: The hemisphere in which the place where Kruger National Park is located lies is the  
> model's next token: ` Southern` (expected `Southern`)  
> probe score (a−b direction): -482.56  |  J-sum South Korea: 25.72 vs South Africa: 35.29  
> J-lens top-8 @L60: [' South', ' Southern', ' south', ' southern', '南', 'South', ' SOUTH', 'Southern']  
> template top-6 @L60: [('hemisphere', 0.231), ('asia', 0.1), ('northern', 0.091), ('pacific', 0.084), ('north', 0.081), ('southern', 0.08)]

**South Africa** — route `language`  
> Q: What language is mainly spoken in the place known for Kruger National Park? A:  
> model's next token: `

` (expected `Zulu`)  
> probe score (a−b direction): -667.61  |  J-sum South Korea: 8.51 vs South Africa: 14.28  
> J-lens top-8 @L60: [' English', 'English', '英语', '\n\n', ' english', 'english', ' Afrika', ' Portuguese']  
> template top-6 @L60: [('indonesian', 0.177), ('portuguese', 0.165), ('german', 0.152), ('tamil', 0.146), ('hindi', 0.145), ('russian', 0.145)]

**South Africa** — route `continent`  
> Geography note: Kruger National Park can be found on the continent called  
> model's next token: ` Africa` (expected `Africa`)  
> probe score (a−b direction): -539.53  |  J-sum South Korea: 11.94 vs South Africa: 28.55  
> J-lens top-8 @L60: [' Africa', '非洲', '...', ' __', ' African', 'Africa', '____', '?']  
> template top-6 @L60: [('africa', 0.196), ('asia', 0.149), ('continent', 0.133), ('arabia', 0.102), ('zimbabwe', 0.097), ('sudan', 0.096)]


## San Diego vs San Francisco (latent, L62)

**San Diego** — route `second_word_letters`  
> Fact: The number of letters in the second word of the name of the place where the SeaWorld park near Mission Bay is located is  
> model's next token: ` ` (expected `5`)  
> probe score (a−b direction): -32.65  |  J-sum San Diego: -0.35 vs San Francisco: -0.78  
> J-lens top-8 @L62: [' greater', ' even', ' divisible', ' equal', ' ', ' odd', ' less', ' exactly']  
> template top-6 @L62: [('eleven', 0.141), ('twelve', 0.106), ('forty', 0.101), ('nine', 0.089), ('nineteen', 0.088), ('seven', 0.082)]

**San Diego** — route `continent`  
> Fact: The continent containing the place where Balboa Park is located is  
> model's next token: ` North` (expected `North America`)  
> probe score (a−b direction): +5.69  |  J-sum San Diego: 9.69 vs San Francisco: 9.01  
> J-lens top-8 @L62: [' North', ' called', 'North', ' NA', ' located', ' known', ' named', ' the']  
> template top-6 @L62: [('asia', 0.118), ('continent', 0.093), ('africa', 0.091), ('russia', 0.067), ('arabia', 0.059), ('panama', 0.056)]

**San Diego** — route `hemisphere`  
> Fact: The hemisphere in which the place where Comic-Con is located lies is the  
> model's next token: ` Northern` (expected `Northern`)  
> probe score (a−b direction): -27.59  |  J-sum San Diego: 6.91 vs San Francisco: 8.11  
> J-lens top-8 @L62: [' Northern', ' northern', ' Southern', ' same', ' Western', ' southern', 'Northern', ' hemisphere']  
> template top-6 @L62: [('hemisphere', 0.228), ('americas', 0.087), ('pacific', 0.086), ('globe', 0.079), ('northern', 0.071), ('east', 0.06)]

**San Francisco** — route `hemisphere`  
> The place associated with Fisherman's Wharf is in the  
> model's next token: ` city` (expected `Northern`)  
> probe score (a−b direction): -560.89  |  J-sum San Diego: 10.55 vs San Francisco: 15.45  
> J-lens top-8 @L62: [' heart', ' state', ' city', ' __', ' vicinity', ' ___', ' United', ' area']  
> template top-6 @L62: [('oakland', 0.128), ('Golden Gate Bridge', 0.11), ('bronx', 0.105), ('district', 0.103), ('northeastern', 0.089), ('united states', 0.079)]

**San Francisco** — route `continent`  
> Fact: The continent containing the place where the Golden Gate Bridge is located is  
> model's next token: ` North` (expected `North America`)  
> probe score (a−b direction): -170.62  |  J-sum San Diego: 8.78 vs San Francisco: 9.35  
> J-lens top-8 @L62: [' North', ' called', ' located', ' known', ' larger', 'North', ' also', ' named']  
> template top-6 @L62: [('asia', 0.118), ('continent', 0.094), ('africa', 0.08), ('russia', 0.078), ('located', 0.062), ('california', 0.061)]

**San Francisco** — route `hemisphere`  
> Geography note: Alcatraz Island is located in the  
> model's next token: ` San` (expected `Northern`)  
> probe score (a−b direction): -650.96  |  J-sum San Diego: 18.82 vs San Francisco: 24.98  
> J-lens top-8 @L62: [' middle', ' San', ' bay', ' midst', ' Golden', ' center', ' heart', 'San']  
> template top-6 @L62: [('gulf', 0.205), ('Golden Gate Bridge', 0.179), ('bay', 0.148), ('oakland', 0.144), ('strait', 0.135), ('pacific', 0.132)]


## United States vs United Nations (latent, L52)

**United States** — route `currency`  
> Fact: The currency used in the place where the Grand Canyon is located is the  
> model's next token: ` US` (expected `dollar`)  
> probe score (a−b direction): +448.18  |  J-sum United States: 10.05 vs United Nations: 9.34  
> J-lens top-8 @L52: [' currency', 'currency', ' United', ' Currency', ' currencies', ' American', 'United', '货币']  
> template top-6 @L52: [('euro', 0.181), ('yuan', 0.163), ('sterling', 0.145), ('currency', 0.134), ('dollar', 0.115), ('fiat', 0.112)]

**United States** — route `language`  
> Travel note: around the U.S. Congress, the main language you will hear is  
> model's next token: ` English` (expected `English`)  
> probe score (a−b direction): -140.82  |  J-sum United States: 2.17 vs United Nations: 1.49  
> J-lens top-8 @L52: [' language', ' languages', '的语言', 'language', '语言', '普通话', '汉语', ' lingua']  
> template top-6 @L52: [('tamil', 0.167), ('portuguese', 0.167), ('german', 0.161), ('indonesian', 0.154), ('filipino', 0.151), ('korean', 0.147)]

**United States** — route `continent`  
> Geography note: the Fourth of July holiday can be found on the continent called  
> model's next token: ` North` (expected `North America`)  
> probe score (a−b direction): +673.41  |  J-sum United States: 6.32 vs United Nations: 7.12  
> J-lens top-8 @L52: ['____', ' __', '...', ' ______', ' ____', '__', '…', '___']  
> template top-6 @L52: [('continent', 0.132), ('asia', 0.106), ('africa', 0.086), ('arabia', 0.08), ('european', 0.073), ('poland', 0.073)]

**United Nations** — route `hemisphere`  
> The place associated with peacekeeping blue helmets is in the  
> model's next token: ` heart` (expected `Northern`)  
> probe score (a−b direction): -477.10  |  J-sum United States: 6.32 vs United Nations: 5.96  
> J-lens top-8 @L52: ['____', ' vicinity', ' area', ' region', ' world', ' country', ' countries', ' __']  
> template top-6 @L52: [('hague', 0.132), ('netherlands', 0.088), ('philippines', 0.087), ('geneva', 0.081), ('philippine', 0.08), ('united nations', 0.079)]

**United Nations** — route `language`  
> Fact: The language most widely spoken in the place where the Secretary-General is located is  
> model's next token: ` French` (expected `English`)  
> probe score (a−b direction): -263.13  |  J-sum United States: 2.04 vs United Nations: 2.23  
> J-lens top-8 @L52: [' languages', ' language', '的语言', '语言', 'language', ' Languages', ' lingua', 'Language']  
> template top-6 @L52: [('tamil', 0.171), ('german', 0.163), ('russian', 0.163), ('indonesian', 0.16), ('turkish', 0.158), ('czech', 0.149)]

**United Nations** — route `second_word_letters`  
> Q: How many letters are in the second word of the name of the place known for the Secretary-General? A:  
> model's next token: `<|im_end|>` (expected `7`)  
> probe score (a−b direction): -239.35  |  J-sum United States: 1.57 vs United Nations: 3.69  
> J-lens top-8 @L52: ['<think>', ' answer', ' Answer', 'Answer', '<|im_end|>', '\n\n', '答案', '的答案']  
> template top-6 @L52: [('historian', 0.054), ('surrender', 0.052), ('subsidiary', 0.049), ('modifications', 0.046), ('noah', 0.045), ('mans', 0.045)]


## Emission-natural examples (text ends right before the phrase)

**New Zealand** — `…lines was Jensen.…
View this post

DPA Microphones has announced the appointment of Direct Imports as its distributor in`  
> model's next token: ` the`; log p(first token): -3.60  
> J-lens top-8 @L62: [' the', ' Greater', ' New', ' Central', ' Great', ' sub', ' Israel', ' North']  
> template top-6 @L62: [('sweden', 0.108), ('New Zealand', 0.104), ('poland', 0.095), ('argentina', 0.089), ('romania', 0.084), ('austria', 0.081)]

**New Zealand** — `… full stock of DPA products for live, recording and broadcast applications. “We are delighted to have been appointed the`  
> model's next token: ` exclusive`; log p(first token): -2.35  
> J-lens top-8 @L62: [' exclusive', ' official', ' new', ' sole', ' New', '…', ' D', ' distributor']  
> template top-6 @L62: [('distributors', 0.128), ('sponsor', 0.1), ('distributor', 0.098), ('dealer', 0.095), ('New Zealand', 0.079), ('dealers', 0.078)]

**San Antonio** — `…eal that makes them happy, and I enjoy giving fifteen percent of every commission to a nonprofit of their choice here in`  
> model's next token: ` the`; log p(first token): -2.33  
> J-lens top-8 @L60: [' San', 'San', ' Greater', ' DC', ' Austin', ' greater', ' Santa', ' metro']  
> template top-6 @L60: [('dallas', 0.145), ('austin', 0.132), ('cincinnati', 0.131), ('california', 0.116), ('omaha', 0.115), ('pittsburgh', 0.109)]

**San Antonio** — `…and has been in real estate now for five years, one year with Crowned Eagle Realty. She has an insiders knowledge of the`  
> model's next token: ` local`; log p(first token): -5.36  
> J-lens top-8 @L60: [' area', ' market', ' areas', ' markets', ' local', ' Market', 'area', 'market']  
> template top-6 @L60: [('calgary', 0.104), ('market', 0.103), ('ontario', 0.085), ('suburbs', 0.083), ('neighborhood', 0.08), ('metropolitan', 0.078)]
