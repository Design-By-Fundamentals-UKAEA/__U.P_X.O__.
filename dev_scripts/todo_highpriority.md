# LOW HANGING FRUITS -- 13-05-2024 : 19-05-2024
# LOW HANGING FRUITS -- 20-05-2024 : 26-05-2024
# LOW HANGING FRUITS -- 26-05-2024 : 02-05-2024
# LOW HANGING FRUITS -- 03-05-2024 : 09-05-2024
REVISITS
--------
. Revisit MCGS2d ( DONE )
. Revisit MCGS2d synth.TGT:synth.SMP ( DONE )
. Revisit MCGS3d (  )

PRE-REQUISITES
--------------
. pre-requite: subset-extraction ( DONE )
. UPXO GS storage and retrieval - raw text file

GRAIN STRUCTURE CHARACTERISATION
--------------------------------
. nth order neighbour identification: MCGS2D ( DONE )
. neighbour identification: MCGS3D (  )
. nth order neighbour identification: MCGS3D (  )
. use neighbours and cluster the grains in mcgs -- number of grains based
. use neighbours and cluster the grains in mcgs -- grain area distribution based

GRAIN STRUCTURE OPERATIONS
--------------------------
. Implement scikitimage.remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None)  ( low priority. Not DONE )
. Implement scikitimage.remove_small_objects(ar, min_size=64, connectivity=1, *, out=None)  ( low priority. Not DONE )

REPRESENTATIVENESS QUALIFICATION
--------------------------------
. subset-main set representativeness qualification check: rel entropy -- MCGS 2D ( DONE )
. Representativeness field generation and use in representativeness assessment.
    . Gen-1a: O(n) self-representativeness field estimation - MCGS 2D ( DONE ): number of neighbours
    . Gen-1b: O(n) self-representativeness field estimation - MCGS 2D ( DONE ): Grain area
    . Gen-2: O(n) R-field comparisons for parameter-wise representativeness assessment of a MCGS2D target - MCGS2D sample pair: Identical domain size.
    . Gen-3: O(n) R-field comparisons for representativeness assessment of a MCGS2D target - MCGS2D sample pair: Non-Identical domain size: nneigh.
. subset-main set representativeness qualification check -- MCGS 3D (  )
. parent MCGS3D -- 2D slice GS representativeness qualification -- VTGS3D voxelated ( WORK IN PROGRESS )
. MTEX - pending work -- texture representativeness (  )
. MTEX - pending work -- morphology representativeness (  )

RESEARCH QUESTIONS
------------------
    1. Are npixels-grain and grain area correlated?
    2. Are npixels-grain and npixels-grain boudnary correlated?
    3. Are npixels-grain and eq.diameter correlated?
    4. What minimum number of domain-complete grains would be needed to ensure representativeness?
    5. What minimum number of domain-internal grains would be needed to ensure representativeness?
    6. We have a representativeness metric attached to a grain structure. But can we have it as a representativeness field overlaid on the grain structure?
        ANSWER: Yes.

. CASE STUDY - 1: NxN MCGS2D synth.T : synth.S repr assessment.
    A. Border grains included, Single pixel grains removed, Outlier data points removed.
        A. 100x100, Q=32, alg202a, tslices=[2, 4, 10, 15, 20, 30, 40, 48]. ( DONE )
        B. 200x200, Q=32, alg202a, tslices=[2, 4, 10, 15, 20, 30, 40, 48]. ( DONE )
        C. 500x500, Q=32, alg202a, tslices=[2, 4, 10, 15, 20, 30, 40, 48]. ( DONE )




Twin and block type feature generation has been integrated into the mainstream UPXO i nboth 2D and 3D.

Multi-parameter representativeness assessment of 2D and 3D grain structures.

Size dependent repr testing in 2D and 3D - now in mainstream UPXO.

TECH - 1:
STEP 1 A: Pipeline for Surface --> Sub-surface feature correlation studies has been set-up.
STEP - 1 B: 3D MCGS --> 2D slice represemtativeness assessment.

STEP 2: Generation

TECH - 2:
"""

shrI vijayadAsarU: payaNAda daari deepa

jee vanadalli jayakkAgi guriyirabeku
ಜೀವನದಲ್ಲಿ ಜಯಕ್ಕಾಗಿ ಗುರುವಿರಬೇಕು.

Guri muttalu gurugala anugrahaviddu avara gulAmanAgabeku.
ಗುರಿ ಮುಟ್ಟಲು ಗುರುಗಳ ಅನುಗ್ರಹವಿದ್ದು ಅವರ ಗುಲಾಮನಾಗಬೇಕು.

Guri daatalu, aa guruvee guruvAgabeku.
ಗುರಿ ದಾತಲಾಗುರುವೀಗುರುವಾಗಬೇಕು.

gurugala anugraha padeyuvudE sAdhakara moTTamodala guri.
ಗುರುಗಳ ಅನುಗ್ರಹ ಪಡೆಯುವುದೇ ಸಾಧಕರ ಮೊಟ್ಟಮೊದಲ ಗುರಿ

avara anugrahave taapadoLu neraLu.
ಅವರ ಅನುಗ್ರಹವೇ ತಾಪದೊಳು ನೆರಳು

modalige naavu taapadallidEve embo arivaagabEku.
ಮೊದಲಿಗೆ ನಾವು ತಾಪದಲ್ಲಿದ್ದೇವೆ ಎಂಬೋ ಅರಿವಾಗಬೇಕು

bIsuva gALiyali alagaaDuva marada eleyante baLaluttaa naraLuttiddEve embo arivAgabeku.
ಬೀಸುವ ಗಾಳಿಯಲಿ ಅಲಗಾಡುವ ಮರದ ಎಲೆಯಂತೆ ಬಳಲುತ್ತಾ ನರಳುತ್ತಿದ್ದೇವೆ ಎಂಬೋ ಅರಿವಾಗಬೇಕು

Aaga taane, neraLigAgi mattu jeevanadallirabEkAda dishegaagi huDukaaTa mattu kaatarateya munnaDeya naDemODItu.
ಆಗ ತಾನೇ, ನೆರಳಿಗಾಗಿ ಮತ್ತು ಜೀವನದಲ್ಲಿರಬೇಕಾದ ದಿಶೆಗಾಗಿ ಹುಡುಕಾಟ ಮತ್ತು ಕಾತರತೆಯ ಮುನ್ನಡೆಯ ನಡೆಮೂಡಿತು.

illade hOdare badukella bari maraLubhoomiyali maraLa mEle neerannarisidantAdItu.
ಇಲ್ಲದೆ ಹೋದರೆ ಬದುಕೆಲ್ಲ ಬರಿ ಮರಳುಭೂಮಿಯಲಿ ಮರಳ ಮೇಲೆ ನೀರನ್ನರಿಸಿದಂತಾದೀತು.

aa munnaDeya nantaravE iruvudavara neraLu.
ಆ ಮುನ್ನಡೆಯ ನಂತರವೇ ಇರುವುದರ ನೆರಳು.

neraLe anugraha, anugrahadinda dishe, disheya Adiyali guru, hAdiyalliyU guru.
ನೇರಳೆ ಅನುಗ್ರಹ, ಅನುಗ್ರಹದಿಂದ ದಿಶೆ, ದಿಶೆಯ ಅಡಿಯಲ್ಲಿ ಗುರು, ಹಾದಿಯಲ್ಲಿಯೂ ಗುರು, ಒಳಗು ಗುರು, ಹೊರಗೂ ಗುರು. ಅಷ್ಟೇ ಅಲ್ಲ, ಗುರುವಿನಡಿಯಲ್ಲಿ ದಿಶೆ.

AdiyiMda Adyantarahitanatta namma nimmellara payaNadAraMbha.
ಅಡಿಯಿಂದ ಅಡ್ಯಂತಾರಹಿತನತ್ತ ನಮ್ಮ ನಿಮ್ಮೆಲ್ಲರ ಪಯಣದಾರಂಭ.

ee neraLu bittAga biduva, bEroDeyada yaavudO ondu marada neraLantalla.
ಈ ನೆರಳು ಬಿಟ್ಟಾಗ ಬಿಡುವ, ಬೇರೊಡೆಯದ ಯಾವುದೋ ಒಂದು ಮರದ ನೆರಳಂತಲ್ಲ.

idu bidisuva neraLu. bittaga iruva neraLu. beLeva neraLu. ALakkiLiva neraLu. ALada neraLu. tampAda neraLu. suDuva neraLu.
ಇದು ಬಿಡಿಸುವ ನೆರಳು. ಬಿಟ್ಟಾಗಲು ಇರುವ ನೆರಳು. ಬೆಳೆವ ನೆರಳು. ಆಳಕ್ಕಿಳಿವ ನೆರಳು. ಆಲದ ನೆರಳು. ತಂಪಾದ ನೆರಳು. ಸುಡುವ ನೆರಳೂ ಹೌದು.

ee neraLalliddare kaTTiTTa jaya.
ಈ ನೆರಳಲ್ಲಿದ್ದವಗೆ ಕಟ್ಟಿಟ್ಟ ಜಯ.

iMtaha guru kai hiDidu mElettidare vijaya.
ಇಂತಹ ಗುರು ಕೈ ಹಿಡಿದು ಮೇಲೆತ್ತಿದರೆ ವಿಜಯ.

aa ajayana kaaNalee vijayavirabEku.
ಆ ಅಜಯನ ಕಾಣಲೀ ವಿಜಯವಿರಬೇಕು.

ayanajayagaLalliddU iradaMte sompAgi OtaprotavAgi beLedu haasi
hokkiruva bErpaDisalAgadaMtiruva A purANapurushana, A vEdapurushana A
parampurushana viLAsavanu, jnaana bhakti haagu vairAgyagaLa samatOlanada
bunAdiyataa bitti tOruva gurugaLe namma Shri VijayadAsa gurugaLu.
ಅಯನಜಯಗಳಲ್ಲಿದೂ ಇರದಂತೆ ಸೊಂಪಾಗಿ ಓತಪ್ರೋತವಾಗಿ ಬೆಳೆದು ಹಾಸಿ
ಹೊಕ್ಕಿರುವ ಬೇರ್ಪಡಿಸಲಾಗದಂತಿರುವ ಆ ಪುರಾಣಪುರುಷನ, ಆ ವೇದಪುರುಷನ ಆ
ಪರಂಪುರುಷನ ವಿಳಾಸವನು, ಜ್ಞಾನ ಭಕ್ತಿ ಹಾಗು ವೈರಾಗ್ಯಗಳ ಸಮತೋಲನದ
ಬುನಾದಿಯತಾ ಬಿತ್ತಿ ತೋರುವ ಗುರುಗಳೇ ನಮ್ಮ ಶ್ರೀ ವಿಜಯದಾಸ ಗುರುಗಳು.

ivarillade guriya giriyilla, guriya guritilla.
ಇವರಿಲ್ಲದೆ ಗುರಿಯ ಗಿರಿಯಿಲ್ಲ, ಗುರಿಯ ಗುರಿತಿಲ್ಲ.

sulabhavAgi A guruvu ee guruvAgatakkavarivaru.
ಸುಲಭವಾಗಿ ಆ ಗುರುವು ಈ ಗುರುವಾಗತಕ್ಕವರಿವರು.

ivara neraLallidavage mikkella aa gurugaLu vAtsalyadiMda samIparAgi namma
jeevOdhArakke AtModhArakke kAraNarAgi ee gurugaLAguttare.
ಇವರ ನೆರಳಲಿದವಗೆ ಮಿಕ್ಕೆಲ್ಲ ಆ ಗುರುಗಳು ವಾತ್ಸಲ್ಯದಿಂದ ಸಮೀಪರಾಗಿ ನಮ್ಮ
ಜೀವೋಧರಕ್ಕೆ ಆತ್ಮೋಧಾರಕ್ಕೆ ಕಾರಣರಾಗಿ ಈ ಗುರುಗಳಾಗುತ್ತಾರೆ.

intavara sEve biDabAradu, yaakeMdare sEveyE sAdhakanige gurugaLa neraLatta hogalu bEkAda
shakti.
ಇಂತವರ ಸೇವೆ ಬಿಡಬಾರದು, ಯಾಕೆಂದರೆ ಸೇವೆಯೇ ಸಾಧಕನಿಗೆ ಗುರುಗಳ ನೆರಳತ್ತ ಹೋಗಲು ಬೇಕಾದ ಶಕ್ತಿ ಮಾತು ಮಾರ್ಗ ಕೂಡ.

avara kAruNyada kaDalalli krupeya alegaLigeNeyilla.
ಅವರ ಕಾರುಣ್ಯದ ಕಡಲಲ್ಲಿ ಕೃಪೆಯ ಅಲೆಗಳಿಗೆಣೆಯಿಲ್ಲ.

intavarannu naa biDalaare. ivarannu smarisi badukabEku. smarisalu badukabEku.
badukalu smarisabEku. badukidare smarisabEku. smaraNeya badukAgabEku. smaraneyE
badukAgabEku.
ಇಂತವರನ್ನು ನಾ ಬಿಡಲಾರೆ. ಇವರನ್ನು ಸ್ಮರಿಸಿ ಬದುಕಬೇಕು. ಸ್ಮರಿಸಲು ಬದುಕಬೇಕು.
ಬದುಕಲು ಸ್ಮರಿಸಬೇಕು. ಬದುಕಿದರೆ ಸ್ಮರಿಸಬೇಕು. ಸ್ಮರಣೆಯ ಬದುಕಾಗಬೇಕು. ಸ್ಮರಣೆಯೇ
ಬದುಕಾಗಬೇಕು.

nAnu nAneMdu kuNiva mithyAhaMkArakke guru smaraNeyindalE konemADi
nijAhankArada beLaka kANabEku.
ನಾನು ನಾನೆಂದು ಕುಣಿವ ಮಿಥ್ಯಾಹಂಕಾರಕ್ಕೆ ಗುರು ಸ್ಮರಣೆಯಿಂದಲೇ ಕೊನೆಮಾಡಿ ನಿಜಹಂಕಾರದ ಬೆಳಕ ಕಾಣಬೇಕು.

kANalavara charaNakeragabEku charaNagaLaDiyirabEku. AgAguvudu beLaku.
ಕಾಣಲಾವರ ಚರಣಕೆರಗಬೇಕು ಚರಣಗಳಡಿಯಿರಬೇಕು. ಅಗಾಗುವುದು ದಿವ್ಯಬೆಳಕಿನಾನುಭವ.

namage beLakAdare tAne naanu nammoLage nAvAgi niMtu kai mugidu
talebAgi vEMkaTEshanige suprabhAta hADalu sAdhya.
ನಮಗೆ ಬೆಳಕಾದರೆ ತಾನೇ ನಾನು ನಮ್ಮೊಳಗೇ ನಾವಾಗಿ ನಿಂತು ಕೈ ಮುಗಿದು ತಲೆಬಾಗಿ ವೆಂಕಟೇಶನಿಗೆ ಸುಪ್ರಭಾತ ಹಾಡಲು ಸಧ್ಯ.

illade hOdare elleDeyu kuruDisuva kaggattalAvarisItu.
ಇಲ್ಲದೆ ಹೋದರೆ ಎಲ್ಲೆಡೆಯೂ ಕುರುಡಿಸುವ ಕಗ್ಗತ್ತಲವರಿಸೀತು.

I aMtarsUryOdayakke kAraNa guru vijayadAsaru.
ಈ ಅಂತರ್ಸೂರ್ಯೋದಯಕ್ಕೆ ಕರಣರೇ ನಮ್ಮ ಗುರು ವಿಜಯದಾಸರು.

kuruDisuva jyOtiswarUpana taMpAda atitIkShNavAda hoMgiraNagaLa taNisuNIsuva ella gurugaLa
kAruNyakEMdragaLige mukhyadwAravE guru vijayadAsa gurugaLu.
ಕುರುಡಿಸುವ ಜ್ಯೋತಿಸ್ವರೂಪನಾ ತಂಪಾದ ಅತಿತಿಕ್ಷಣವಾದ ಹೊಂಗಿರಣಗಳ ತಣಿಸುಣಿಸುವ ಎಲ್ಲ ಗುರುಗಳ ಕಾರುಣ್ಯಕೇಂದ್ರಗಳಿಗೆ ಮುಖ್ಯದ್ವಾರ ನಮ್ಮ ಗುರು ವಿಜಯದಾಸರು.

oLitellavudake kAraNa shree. shrIkArakke kAraNanane shRI hari.
ಒಳಿತೆಲ್ಲವುದಕೆ ಕರಣ ಶ್ರೀ. ಶ್ರೀಕಾರಕ್ಕೆ ಆಧಾರನೇ ಶ್ರೀ ಹರಿ.

I shRi harige kai mugidu guru vijayadAsara kavachavannamagIyda guruvu shrI vyAsa viTTala dAsaru.
ಈ ಶ್ರೀ ಹರಿಗೆ ಕೈ ಮುಗಿದು ಗುರು ವಿಜಯದಾಸರ ಕವಚವನ್ನಾಮಗೀಯ್ದ ಗುರುವು ಶ್ರೀ ವ್ಯಾಸ  ವಿಠ್ಠಲ ದಾಸರು.

"""



ಜೀವನದಲ್ಲಿ ಜಯಕ್ಕಾಗಿ ಗುರುವಿರಬೇಕು. ಗುರಿ ಮುಟ್ಟಲು ಗುರುಗಳ ಅನುಗ್ರಹವಿದ್ದು ಅವರ ಗುಲಾಮನಾಗಬೇಕು.

ಗುರಿ ದಾತಲಾಗುರುವೀಗುರುವಾಗಬೇಕು.

ಗುರುಗಳ ಅನುಗ್ರಹ ಪಡೆಯುವುದೇ ಸಾಧಕರ ಮೊಟ್ಟಮೊದಲ ಗುರಿ

ಅವರ ಅನುಗ್ರಹವೇ ತಾಪದೊಳು ನೆರಳು

ಮೊದಲಿಗೆ ನಾವು ತಾಪದಲ್ಲಿದ್ದೇವೆ ಎಂಬೋ ಅರಿವಾಗಬೇಕು

ಬೀಸುವ ಗಾಳಿಯಲಿ ಅಲಗಾಡುವ ಮರದ ಎಲೆಯಂತೆ ಬಳಲುತ್ತಾ ನರಳುತ್ತಿದ್ದೇವೆ ಎಂಬೋ ಅರಿವಾಗಬೇಕು

ಆಗ ತಾನೇ, ನೆರಳಿಗಾಗಿ ಮತ್ತು ಜೀವನದಲ್ಲಿರಬೇಕಾದ ದಿಶೆಗಾಗಿ ಹುಡುಕಾಟ ಮತ್ತು ಕಾತರತೆಯ ಮುನ್ನಡೆಯ ನಡೆಮೂಡಿತು.

ಇಲ್ಲದೆ ಹೋದರೆ ಬದುಕೆಲ್ಲ ಬರಿ ಮರಳುಭೂಮಿಯಲಿ ಮರಳ ಮೇಲೆ ನೀರನ್ನರಿಸಿದಂತಾದೀತು.

ಆ ಮುನ್ನಡೆಯ ನಂತರವೇ ಇರುವುದರ ನೆರಳು.

ನೇರಳೆ ಅನುಗ್ರಹ, ಅನುಗ್ರಹದಿಂದ ದಿಶೆ, ದಿಶೆಯ ಅಡಿಯಲ್ಲಿ ಗುರು, ಹಾದಿಯಲ್ಲಿಯೂ ಗುರು, ಒಳಗು ಗುರು, ಹೊರಗೂ ಗುರು. ಅಷ್ಟೇ ಅಲ್ಲ, ಗುರುವಿನಡಿಯಲ್ಲಿ ದಿಶೆ.

ಅಡಿಯಿಂದ ಅಡ್ಯಂತಾರಹಿತನತ್ತ ನಮ್ಮ ನಿಮ್ಮೆಲ್ಲರ ಪಯಣದಾರಂಭ.

ಈ ನೆರಳು ಬಿಟ್ಟಾಗ ಬಿಡುವ, ಬೇರೊಡೆಯದ ಯಾವುದೋ ಒಂದು ಮರದ ನೆರಳಂತಲ್ಲ.

ಇದು ಬಿಡಿಸುವ ನೆರಳು. ಬಿಟ್ಟಾಗಲು ಇರುವ ನೆರಳು. ಬೆಳೆವ ನೆರಳು. ಆಳಕ್ಕಿಳಿವ ನೆರಳು. ಆಲದ ನೆರಳು. ತಂಪಾದ ನೆರಳು. ಸುಡುವ ನೆರಳೂ ಹೌದು.

ಈ ನೆರಳಲ್ಲಿದ್ದವಗೆ ಕಟ್ಟಿಟ್ಟ ಜಯ.

ಇಂತಹ ಗುರು ಕೈ ಹಿಡಿದು ಮೇಲೆತ್ತಿದರೆ ವಿಜಯ.

ಆ ಅಜಯನ ಕಾಣಲೀ ವಿಜಯವಿರಬೇಕು.

ಅಯನಜಯಗಳಲ್ಲಿದೂ ಇರದಂತೆ ಸೊಂಪಾಗಿ ಓತಪ್ರೋತವಾಗಿ ಬೆಳೆದು ಹಾಸಿ
ಹೊಕ್ಕಿರುವ ಬೇರ್ಪಡಿಸಲಾಗದಂತಿರುವ ಆ ಪುರಾಣಪುರುಷನ, ಆ ವೇದಪುರುಷನ ಆ
ಪರಂಪುರುಷನ ವಿಳಾಸವನು, ಜ್ಞಾನ ಭಕ್ತಿ ಹಾಗು ವೈರಾಗ್ಯಗಳ ಸಮತೋಲನದ
ಬುನಾದಿಯತಾ ಬಿತ್ತಿ ತೋರುವ ಗುರುಗಳೇ ನಮ್ಮ ಶ್ರೀ ವಿಜಯದಾಸ ಗುರುಗಳು.

ಇವರಿಲ್ಲದೆ ಗುರಿಯ ಗಿರಿಯಿಲ್ಲ, ಗುರಿಯ ಗುರಿತಿಲ್ಲ.

ಸುಲಭವಾಗಿ ಆ ಗುರುವು ಈ ಗುರುವಾಗತಕ್ಕವರಿವರು.

ಇವರ ನೆರಳಲಿದವಗೆ ಮಿಕ್ಕೆಲ್ಲ ಆ ಗುರುಗಳು ವಾತ್ಸಲ್ಯದಿಂದ ಸಮೀಪರಾಗಿ ನಮ್ಮ
ಜೀವೋಧರಕ್ಕೆ ಆತ್ಮೋಧಾರಕ್ಕೆ ಕಾರಣರಾಗಿ ಈ ಗುರುಗಳಾಗುತ್ತಾರೆ.

ಇಂತವರ ಸೇವೆ ಬಿಡಬಾರದು, ಯಾಕೆಂದರೆ ಸೇವೆಯೇ ಸಾಧಕನಿಗೆ ಗುರುಗಳ ನೆರಳತ್ತ ಹೋಗಲು ಬೇಕಾದ ಶಕ್ತಿ ಮಾತು ಮಾರ್ಗ ಕೂಡ.

ಅವರ ಕಾರುಣ್ಯದ ಕಡಲಲ್ಲಿ ಕೃಪೆಯ ಅಲೆಗಳಿಗೆಣೆಯಿಲ್ಲ.

ಇಂತವರನ್ನು ನಾ ಬಿಡಲಾರೆ. ಇವರನ್ನು ಸ್ಮರಿಸಿ ಬದುಕಬೇಕು. ಸ್ಮರಿಸಲು ಬದುಕಬೇಕು.
ಬದುಕಲು ಸ್ಮರಿಸಬೇಕು. ಬದುಕಿದರೆ ಸ್ಮರಿಸಬೇಕು. ಸ್ಮರಣೆಯ ಬದುಕಾಗಬೇಕು. ಸ್ಮರಣೆಯೇ
ಬದುಕಾಗಬೇಕು.

ನಾನು ನಾನೆಂದು ಕುಣಿವ ಮಿಥ್ಯಾಹಂಕಾರಕ್ಕೆ ಗುರು ಸ್ಮರಣೆಯಿಂದಲೇ ಕೊನೆಮಾಡಿ ನಿಜಹಂಕಾರದ ಬೆಳಕ ಕಾಣಬೇಕು.

ಕಾಣಲಾವರ ಚರಣಕೆರಗಬೇಕು ಚರಣಗಳಡಿಯಿರಬೇಕು. ಅಗಾಗುವುದು ದಿವ್ಯಬೆಳಕಿನಾನುಭವ.

ನಮಗೆ ಬೆಳಕಾದರೆ ತಾನೇ ನಾನು ನಮ್ಮೊಳಗೇ ನಾವಾಗಿ ನಿಂತು ಕೈ ಮುಗಿದು ತಲೆಬಾಗಿ ವೆಂಕಟೇಶನಿಗೆ ಸುಪ್ರಭಾತ ಹಾಡಲು ಸಧ್ಯ.

ಇಲ್ಲದೆ ಹೋದರೆ ಎಲ್ಲೆಡೆಯೂ ಕುರುಡಿಸುವ ಕಗ್ಗತ್ತಲವರಿಸೀತು.

ಈ ಅಂತರ್ಸೂರ್ಯೋದಯಕ್ಕೆ ಕರಣರೇ ನಮ್ಮ ಗುರು ವಿಜಯದಾಸರು.

ಕುರುಡಿಸುವ ಜ್ಯೋತಿಸ್ವರೂಪನಾ ತಂಪಾದ ಅತಿತಿಕ್ಷಣವಾದ ಹೊಂಗಿರಣಗಳ ತಣಿಸುಣಿಸುವ ಎಲ್ಲ ಗುರುಗಳ ಕಾರುಣ್ಯಕೇಂದ್ರಗಳಿಗೆ ಮುಖ್ಯದ್ವಾರ ನಮ್ಮ ಗುರು ವಿಜಯದಾಸರು.

ಒಳಿತೆಲ್ಲವುದಕೆ ಕರಣ ಶ್ರೀ. ಶ್ರೀಕಾರಕ್ಕೆ ಆಧಾರನೇ ಶ್ರೀ ಹರಿ.

ಈ ಶ್ರೀ ಹರಿಗೆ ಕೈ ಮುಗಿದು ಗುರು ವಿಜಯದಾಸರ ಕವಚವನ್ನಾಮಗೀಯ್ದ ಗುರುವು ಶ್ರೀ ವ್ಯಾಸ  ವಿಠ್ಠಲ ದಾಸರು.
