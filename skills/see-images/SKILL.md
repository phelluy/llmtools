---
name: see-images
description: >-
  Lire ou décrire une image (capture d'écran, photo, document scanné) qui n'est
  PAS déjà attachée à la conversation. Si un bloc <image alias="..."> est déjà
  présent, ou si le modèle actif accepte les images, ne pas l'utiliser : la
  description déjà fournie suffit. À charger pour pousser un fichier vers
  l'endpoint Unistra qwen3, recadrer ou agrandir avant analyse, mesurer une
  zone au pixel (analyse pixel déterministe), ou insérer une signature dans un
  PDF scanné.
user-invocable: true
---

# See Images

Décrit des images via le modèle multimodal `qwen3` du provider **unistra**
(`https://conversation.ia.unistra.fr/api`, API compatible OpenAI) en utilisant `curl`.

## Avant tout : l'image est-elle déjà visible ?

Ne pas payer un appel réseau pour une image que la session a déjà lue.

| Situation | Action |
| --- | --- |
| La conversation contient un bloc `<image alias="…">` | C'est **déjà** le contenu de l'image, produit par le describer du harnais. L'exploiter tel quel, ne pas relancer `see.py`. |
| Le modèle actif a `supports_images = true` | Joindre le fichier avec `@chemin.png` suffit : les pixels arrivent directement. |
| Image sur disque, mais rien dans la conversation | Soit `@chemin.png` (si un `[vision_model]` est configuré), soit `see.py` ci-dessous. Les deux donnent une **description texte**. |
| Plusieurs angles, cadrage précis, ou image produite pendant la tâche | `see.py` avec `-p`, et `--crop`/`--scale` plutôt que la capture entière. |

Un `[vision_model]` dans `~/.vibe/config.toml` fait décrire chaque `@image` par
un modèle vision et livre ce texte à la place des pixels ; la description est
générée une seule fois par image et par session :

```bash
grep -A4 '^\[vision_model\]' ~/.vibe/config.toml
```

Conséquence pratique : la description dépend de la consigne de l'utilisateur
pour ce tour. Si elle ne répond pas à la question, relancer `see.py` avec un
prompt ciblé est légitime.

## Script fourni

`see.py`, dans le même dossier que ce `SKILL.md` (les exemples supposent qu'on s'y trouve ; sinon préfixer par le chemin du skill)

```bash
# Description détaillée (défaut, en français)
python3 see.py image.png

# Prompt personnalisé
python3 see.py image.png -p "Que contient ce schéma ?"

# Recadrer puis agrandir avant l'envoi : indispensable sur une capture d'écran
# dont le texte utile est petit (x0,y0,x1,y1 en pixels de l'image source)
python3 see.py shot.png \
  --crop 200,20,700,140 --scale 3 -p "Transcris le titre et le contenu."

# Plusieurs images à la fois
python3 see.py a.png b.png -p "Compare ces deux figures."
```

`--crop` et `--scale` ont besoin de Pillow ; si l'import échoue, relancer avec
`uv run --with pillow python3 see.py …`.

## Configuration

Le script lit :

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `UNISTRA_API_KEY` | (requise) | Clé API du provider `unistra` |
| `SEE_IMAGES_MODEL` | `qwen3` | Identifiant du modèle |
| `SEE_IMAGES_API_BASE` | `https://conversation.ia.unistra.fr/api` | URL de base |

Les valeurs correspondent à la section `[[providers]]` / `[[models]]` de
`~/.vibe/config.toml` (provider `unistra`, modèle `qwen3`).

## Appel curl équivalent (référence)

```bash
python3 - <<'PY'
import base64, json
b64 = base64.b64encode(open("image.png", "rb").read()).decode()
print(json.dumps({
  "model": "qwen3",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Décris cette image en détail."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
  ]}],
}))
PY
# puis envoyer ce JSON via :
# curl -sS -X POST https://conversation.ia.unistra.fr/api/chat/completions \
#   -H "Authorization: Bearer $UNISTRA_API_KEY" \
#   -H "Content-Type: application/json" \
#   --data-binary @payload.json
```

La réponse contient la description dans `choices[0].message.content`.

## Bonnes pratiques

- Préférer le script `see.py` au curl manuel : il gère l'encodage base64,
  le MIME, le JSON et l'extraction de la réponse.
- Vérifier que `UNISTRA_API_KEY` est exporté avant l'appel.
- Formats supportés : png, jpg/jpeg, gif, webp, bmp, tiff.
- Toujours utiliser un timeout : le script applique `--max-time 180`.
- **Recadrer avant d'envoyer.** Une capture 1024x720 envoyée entière perd les
  textes de 12-14 px ; le même contenu recadré et agrandi ×3 est lu sans faute.
  C'est le levier de qualité numéro un, bien avant le choix du prompt.

## Captures d'écran et interfaces graphiques

Deux usages distincts, à ne pas confondre :

- **Sémantique** (« est-ce qu'un dialogue est ouvert ? lequel ? quels boutons,
  dans quel ordre ? ») : le modèle vision est fiable. Formuler la question en
  termes d'état et de contenu, pas de positions.
- **Géométrie** (« à quel pixel cliquer ? ») : le modèle vision n'est **pas**
  fiable, il hallucine des coordonnées cohérentes mais fausses. Mesurer.

Sonde ASCII pour lire la structure d'une capture sans aucun modèle (les
rectangles de widgets ressortent en blocs sombres ; on lit leurs bornes en
lignes/colonnes, donc les coordonnées au pixel près) :

```bash
python3 -c "
from PIL import Image
im = Image.open('shot.png').convert('L').crop((0,0,320,200)).resize((80,40))
ramp = '@%#*+=-:. '
px = im.load()
for y in range(40):
    print(''.join(ramp[min(9,(255-px[x,y])*10//256)] for x in range(80)))
"
```

Sur une sortie 80x40, une ligne `y` vaut `y/40 * hauteur_du_crop` pixels et une
colonne `x` vaut `x/80 * largeur_du_crop`. Exemple vécu : deux rectangles aux
lignes 16-17 sur 40 et colonnes 6-16 / 21-30 dans un crop 320x200 → boutons à
y ≈ 85, x ≈ 24-64 et 84-120 ; clic du premier coup.

Chaîne de test habituelle pour une app GUI native : `Xvfb :99` + `xdotool` +
`scrot`, puis sonde ASCII ou analyse pixel pour la géométrie, et modèle vision
pour confirmer l'état. Toujours recadrer la zone concernée avant de demander une
transcription.

## Localisation précise dans une image (analyse pixel)

**Le modèle vision n'est PAS fiable pour les coordonnées.** Il décrit et
transpose bien le contenu, mais ses rectangles x1,y1,x2,y2 sont
approximatifs et souvent contradictoires d'un appel à l'autre. Ne jamais
s'appuyer sur ses coordonnées pour coller un élément : utiliser l'analyse
pixel déterministe ci-dessous, et réserver `see.py` à la description et à
la vérification finale.

Astuces qui ont fonctionné (via `uv run --with pillow python`) :

- **Profils d'encre par ligne/colonne** : compter les pixels sombres
  (`<160`) par ligne pour trouver les lignes de texte, puis par colonne
  dans chaque bande pour trouver les groupes de mots. Fusionner les
  segments espacés de <25-40 px.
- **Cadres de formulaires** : une case entourée crée une ligne continue et
  fait merger toutes les lignes de texte. Détecter séparément les traits
  (rangées avec >600 px sombres = trait horizontal) et les exclure.
- **Distinguer encre bleue manuscrite / texte imprimé noir** :
  `b - r > 40` (bleu) vs `max(r,g,b) < 170` sans excès de bleu (noir).
  Utile pour ne pas écraser une signature existante.
- **Texte imprimé très clair** (libellés de formulaires type SAP,
  p. ex. « L'agent ») : invisible à seuil 160, visible à seuil `<242`.
  Si une zone paraît vide mais que le formulaire devrait avoir un libellé,
  réessayer avec un seuil plus clair avant de conclure.
- **Fond gris clair vs blanc** : sur un thème clair, une gouttière peut être
  en `248,248,248` et le corps en `255,255,255`. Comparer les couleurs
  exactes plutôt que le seul seuil de luminance.
- **Grille graduée** : dessiner une grille rouge avec étiquettes
  (`ImageDraw`) sur un recadrage aide le modèle vision à situer les
  éléments, mais reste indicatif.

## OCR déterministe (tesseract, optionnel)

`tesseract` n'est **pas installé** sur cette machine (vérifié). S'il est
installé un jour, il donne texte + boîtes de façon déterministe :

```bash
tesseract page.png stdout --psm 11 tsv   # texte + coordonnées par mot
```

En son absence, l'analyse pixel ci-dessus + `see.py` pour la transcription
suffisent.

## Signature d'un PDF scanné (workflow validé)

Workflow qui a fonctionné pour signer un ordre de mission scanné :

1. **Rendre la page à 300 dpi** : `pdftoppm -png -r 300 -f N -l N doc.pdf page`.
2. **Localiser la zone** par analyse pixel (section ci-dessus), pas via le
   modèle vision.
3. **Insérer la signature** : l'image de signature (JPEG blanc/noir) est
   convertie en masque alpha (sombre = opaque), teintée en gris encre avec
   léger bruit, floutée (`GaussianBlur(0.8)`) et légèrement pivotée
   (2-4°) pour se fondre dans le scan :

```python
sig = ImageOps.autocontrast(sig.convert('L')).resize((W, W*71//132), Image.LANCZOS)
alpha = sig.point(lambda v: 0 if v > 200 else min(255, int(255*(200-v)/130)))
# remplir RGBA avec (40±12, 40±12, 50, alpha), puis .rotate(-4, expand=True)
```

4. **Texte manuscrit** (« Strasbourg », date) : police
   `/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf`, jitter
   vertical aléatoire par caractère (±3 px), légère rotation, flou 0.6.
5. **Contrôle avant collage** : compter les pixels en conflit (encre bleue
   ou texte imprimé) dans le rectangle cible sur la page d'origine ; doit
   valoir 0.
6. **Reconstruire le PDF** : `uv run --with img2pdf --with pypdf` :
   `img2pdf.convert(png, layout_fun=img2pdf.get_layout_fun((mm_to_pt(210), mm_to_pt(297))))`
   pour la page modifiée, puis fusion avec `pypdf.PdfWriter` (pages
   originales + page(s) signée(s)).
7. **Vérifier** : `pdftoppm -r 100` sur le PDF final + `see.py` pour
   confirmer la lecture du bloc signé.
