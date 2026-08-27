# buildprompt.md
# GEN AI Project — Antigravity GUI Build Specification

## 0. Purpose of This Prompt

You are building the GUI for the **GEN AI Project**, a university-level Generative AI system whose final goal is to demonstrate a **Prompt-to-Game-Asset Generation workflow** using five major Generative AI model families:

1. Transformer / LLM
2. Autoencoder (AE)
3. Variational Autoencoder (VAE)
4. GAN
5. Diffusion Model

The GUI must make the project look and behave like one coherent application, not like five unrelated ML notebooks glued together.

The application should communicate this pipeline clearly:

```text
USER PROMPT
    ↓
TRANSFORMER / LLM
    ↓
GAME DESIGN SPECIFICATION
    ↓
ASSET GENERATION / PROCESSING
    ├── Diffusion → primary visual generation
    ├── VAE      → controlled visual variations
    ├── AE       → reconstruction / denoising / representation
    └── GAN      → texture / visual refinement
    ↓
FINAL GAME ASSET LIBRARY
```

The GUI is a **demonstration and orchestration layer** over the ML models. It must not pretend that a model is implemented when it is not.

---

# 1. PROJECT GOAL

The final application should allow a user to describe a game idea or game asset in natural language and then move through a controlled generation workflow.

Example:

> "Create a dark fantasy RPG village with a stone house, a wooden sign, a knight NPC, a sword, and a magical blue effect."

The application should make the process understandable:

```text
Natural-language prompt
        ↓
AI understands the game request
        ↓
Structured game/asset plan
        ↓
Visual asset generation
        ↓
Optional reconstruction / variation / refinement
        ↓
Generated assets
        ↓
Asset library / export
```

The application is **not** supposed to be a full game engine.

Do NOT build:

- a playable game engine
- a level editor
- multiplayer
- authentication
- payments
- chat/social features
- unnecessary dashboards
- unrelated AI features
- arbitrary extra ML models
- fake model outputs

The goal is a polished academic demonstration of the project's five Generative AI model families.

---

# 2. MODEL ROLES

The UI must preserve the intended role of each model.

## 2.1 Transformer / LLM

Role:

- understand the user's natural-language request
- generate game design information
- generate character descriptions
- generate NPC dialogue
- generate weapon/environment/prop descriptions
- create structured prompts/specifications for visual generation

The Transformer should be presented as the **reasoning / planning layer**.

Example:

```text
User:
"Create a cursed forest village."

LLM output:

Theme:
Dark fantasy

Environment:
Cursed forest village

Characters:
- Village elder
- Guard
- Witch

Assets:
- Wooden houses
- Broken sign
- Lanterns
- Fog
- Magic symbols
```

Do not make the UI imply that the LLM itself generates the final image.

---

## 2.2 Autoencoder

Role:

- reconstruction
- denoising
- compression / latent representation
- processing uploaded or generated visual assets

Current validated AE experiment:

```text
Dataset:
evilsocket/alucard-sprites

Unique images:
282,511

Image:
128 × 128 × 4 RGBA

Architecture:
1,079,108 parameters

Encoder output:
8 × 8 × 256

Training:
Adam
Learning rate: 0.001
MSE
50 epochs
Global batch size: 64
2 × Tesla T4 during training

Known 10K baseline:
Clean Test MSE ≈ 0.001251
```

The current 280K experiment uses the same validated architecture on the deduplicated 282,511-image dataset.

The GUI should therefore expose AE functionality as something meaningful such as:

```text
Input asset
   ↓
Autoencoder
   ↓
Reconstruction / denoised asset
```

Do not invent numerical performance claims in the GUI.

---

## 2.3 VAE

Role:

- latent-space exploration
- controlled variations
- generation of related game-asset variants

Example:

```text
Base character
      ↓
VAE latent representation
      ↓
Variation controls
      ↓
Character Variant A
Character Variant B
Character Variant C
```

The GUI can show a variation workflow even if the VAE backend is currently a placeholder.

If the model is not connected, clearly label it:

```text
Demo / Backend not connected
```

Never fabricate generated images.

---

## 2.4 GAN

Role:

- texture generation
- visual refinement
- texture/material variation

Example:

```text
Material:
Stone

GAN
 ↓
Stone texture variations
```

The project dataset planning includes texture/material sources such as VastTextures and MatSynth.

The GUI should make GAN feel like a specialized visual-material stage, not a second generic image generator.

---

## 2.5 Diffusion Model

Role:

- primary text-to-image generation
- characters
- environments
- weapons
- props
- effects
- UI assets

This is the main visual generation surface.

Example:

```text
Prompt
 ↓
Diffusion
 ↓
Generated game asset
```

The interface should make it visually prominent.

---

# 3. DESIGN PRINCIPLE

The most important design decision:

## Build one coherent AI studio, not five model pages.

The user should feel:

> "I am designing a game asset with an AI pipeline."

not:

> "I clicked Autoencoder, then another random page, then another random model."

The five model families should appear as stages/modules inside the same product.

---

# 4. VISUAL DIRECTION

Create a modern **dark game-development / AI studio interface**.

The visual language should be:

- dark background
- high contrast
- restrained neon accents
- clean cards
- subtle gradients
- soft shadows
- rounded corners
- strong typography
- game-editor-inspired layout
- professional enough for an academic project demonstration

Do NOT make it look like:

- a generic Bootstrap admin dashboard
- a crypto website
- a children's gaming website
- an overdecorated cyberpunk landing page
- a generic ChatGPT clone

The application should feel like a serious **AI Game Asset Studio**.

---

# 5. COLOR SYSTEM

Use a dark base.

Suggested palette:

```text
Background:
#0B0D12

Surface:
#121620

Elevated Surface:
#181D28

Primary:
#7C5CFF

Secondary:
#21D4FD

Success:
#34D399

Warning:
#FBBF24

Error:
#F87171

Primary text:
#F5F7FA

Secondary text:
#9CA3AF

Border:
#252B38
```

Use accent colors sparingly.

Do not make every card glow.

The UI should look expensive because it is restrained, not because someone discovered CSS box-shadow.

---

# 6. APPLICATION SHELL

Use a persistent application shell.

## Left sidebar

Sections:

```text
GEN AI STUDIO

Dashboard

Create
  ├── Game Concept
  ├── Asset Generator
  └── NPC / Lore

AI Pipeline
  ├── Transformer
  ├── Diffusion
  ├── VAE
  ├── Autoencoder
  └── GAN

Asset Library

Experiments

About Project
```

Do not make every item a separate unrelated application.

The currently active section must be visually obvious.

---

# 7. TOP BAR

Top bar should contain:

Left:

```text
GEN AI STUDIO
```

Center/right:

```text
Current Project: Untitled Game
```

Right:

```text
Model Status
GPU / Backend Status
```

Use compact status indicators.

Example:

```text
● Backend Ready
```

If a backend is not connected:

```text
○ Demo Mode
```

Do not claim GPU availability unless it is actually detected.

---

# 8. DASHBOARD

The dashboard is the first screen.

Purpose:

Give a fast explanation of the project and provide entry points into the workflow.

## Hero section

Title:

```text
Turn Game Ideas Into Game Assets
```

Subtitle:

```text
A multi-model Generative AI pipeline for designing,
generating, reconstructing, varying, and refining 2D game assets.
```

Primary CTA:

```text
Create New Asset
```

Secondary CTA:

```text
Explore AI Pipeline
```

---

## Pipeline visualization

Show:

```text
Prompt
  ↓
Transformer
  ↓
Game Specification
  ↓
Diffusion
  ↓
VAE
  ↓
AE
  ↓
GAN
  ↓
Asset Library
```

This can be horizontal on desktop and vertical on narrow screens.

Each stage should be clickable.

When a stage is clicked, show:

- model name
- role
- input
- output
- status
- relevant project information

---

# 9. CREATE SCREEN

This is the most important user-facing screen.

Layout:

```text
┌──────────────────────────────────────────────────────────┐
│ Create Game Asset                                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ Prompt                                                   │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Describe the game asset you want to create...         │ │
│ │                                                      │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                          │
│ Asset Type        Style             Generation Mode      │
│ [Character ▼]     [Pixel Art ▼]     [Full Pipeline ▼]   │
│                                                          │
│ [ Generate Design ]                                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

The prompt area should be large.

Include example prompt chips:

```text
Dark fantasy knight
Pixel-art forest village
Magic sword
RPG NPC
Stone dungeon
```

Clicking a chip fills the prompt.

---

# 10. STRUCTURED DESIGN RESULT

After the user submits the prompt, the Transformer stage should produce a structured design specification.

Do not dump a wall of generated text.

Use cards.

Example:

```text
GAME CONCEPT

Theme
Dark Fantasy

Environment
Cursed Forest Village

Characters
3

Weapons
2

Props
7

Visual Style
Pixel Art
```

Then show expandable sections:

```text
Character
Weapon
Environment
Props
NPC Dialogue
Generation Prompts
```

Provide:

```text
Edit Specification
Regenerate Specification
Continue to Visual Generation
```

---

# 11. ASSET GENERATOR SCREEN

The generator should have three columns on desktop.

```text
LEFT                  CENTER                 RIGHT

Prompt / Settings     Generated Preview      Generation Info
                     ┌──────────────┐
Asset Type            │              │
Style                 │   IMAGE      │
Seed                  │              │
                     └──────────────┘
```

The center preview must be the visual focus.

Controls should include only meaningful controls.

Suggested controls:

```text
Asset Type
Visual Style
Aspect Ratio
Number of Variations
Seed
```

Do not add 30 sliders just because sliders exist.

---

# 12. MODEL PIPELINE CONTROL

Provide a compact pipeline selector:

```text
● Transformer
      ↓
● Diffusion
      ↓
○ VAE
      ↓
○ Autoencoder
      ↓
○ GAN
```

Each stage should show:

```text
Connected
Demo
Processing
Completed
Unavailable
```

This lets the user understand which models actually participated.

---

# 13. RESULT VIEW

Generated results should appear as a grid.

Example:

```text
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Asset 1 │ │ Asset 2 │ │ Asset 3 │
│         │ │         │ │         │
└─────────┘ └─────────┘ └─────────┘
```

Each card has:

```text
Asset name
Model used
Generation status
```

Actions:

```text
Open
Compare
Save to Library
Run through AE
Create VAE Variations
Refine Texture
```

Do not create destructive actions without confirmation.

---

# 14. IMAGE COMPARISON SCREEN

This is especially important for the Autoencoder experiment.

Create a dedicated comparison component:

```text
ORIGINAL                 RECONSTRUCTION

┌───────────────┐       ┌───────────────┐
│               │       │               │
│    IMAGE      │       │    IMAGE      │
│               │       │               │
└───────────────┘       └───────────────┘
```

Below:

```text
MSE
PSNR
Image Size
Model
Dataset
```

If metrics are unavailable, show:

```text
Metric unavailable
```

Do not invent values.

---

# 15. AUTOENCODER PAGE

Title:

```text
Autoencoder
```

Subtitle:

```text
Reconstruction, denoising and latent representation
```

Show a simple pipeline:

```text
Input
 ↓
Encoder
 ↓
Latent Representation
 ↓
Decoder
 ↓
Reconstruction
```

Display project facts:

```text
Dataset
282,511 unique images

Input
128 × 128 × 4 RGBA

Encoder Output
8 × 8 × 256

Parameters
1,079,108
```

These are project facts and can be shown directly.

Do not present the first 280K validation result as the final test result.

The first run achieved:

```text
Best validation MSE: 0.0004909808631055057
Best epoch: 32
```

If shown, label it explicitly:

```text
Historical 280K validation result
```

not:

```text
Final test accuracy
```

---

# 16. VAE PAGE

Layout:

```text
VAE

Input Asset
     ↓
Latent Space
     ↓
Variation Controls
     ↓
Generated Variants
```

Main visual element should be a latent-space visualization.

Show a 2D latent plot if the backend supports it.

Otherwise use a clearly marked demo visualization.

Controls:

```text
Variation Strength
Style Similarity
Number of Variants
```

Results:

```text
Variant 1
Variant 2
Variant 3
Variant 4
```

---

# 17. GAN PAGE

Title:

```text
Texture Studio
```

This should visually emphasize materials.

Input:

```text
Material Type
[Stone ▼]
```

Options:

```text
Wood
Stone
Metal
Grass
Brick
Fabric
Rock
```

Pipeline:

```text
Material Prompt
      ↓
GAN
      ↓
Texture Variations
```

Show:

```text
Original / Generated / Refined
```

Do not make GAN look like another generic image generator.

---

# 18. DIFFUSION PAGE

This should be one of the most visually impressive pages.

Large prompt box.

Controls:

```text
Prompt
Negative Prompt
Style
Resolution
Number of Images
Seed
```

Main output:

```text
Generated Images
```

Include prompt metadata below each result.

Example:

```text
Generated with:
Diffusion Model

Prompt:
"Pixel-art knight with..."
```

If the backend is unavailable:

```text
Diffusion backend not connected

UI preview mode
```

Never generate fake "AI" output using static images without labeling it as a demo.

---

# 19. TRANSFORMER PAGE

The Transformer page should emphasize structured game design rather than raw chat.

Layout:

```text
GAME DESIGN INPUT
        ↓
TRANSFORMER
        ↓
STRUCTURED OUTPUT
```

Output cards:

```text
Game Theme
Characters
Environment
Weapons
Props
NPC Dialogue
Visual Prompts
```

Include a collapsible "Raw Model Output" panel for technical demonstration.

This is useful for an academic viva because it shows that the Transformer is doing structured generation rather than simply acting as a chatbot.

---

# 20. ASSET LIBRARY

The asset library should feel like a game-development asset browser.

Grid view:

```text
[image] [image] [image] [image]
Name    Name    Name    Name

[image] [image] [image] [image]
```

Filters:

```text
All
Characters
Weapons
Environments
Props
Textures
NPC
Generated
Processed
```

Each asset should retain metadata:

```text
Name
Type
Source
Model
Created
Prompt
```

The UI should make it easy to compare assets.

---

# 21. EXPERIMENTS PAGE

This page is for the academic / ML side.

Show experiment cards:

```text
Autoencoder 280K
Status: Completed / Running

VAE
Status: Demo / Not Connected

GAN
Status: Demo / Not Connected

Diffusion
Status: Connected / Demo

Transformer
Status: Connected / Demo
```

For the Autoencoder experiment show:

```text
Dataset
282,511 unique images

Train
254,260

Validation
14,126

Test
14,125

Architecture
1,079,108 parameters

Batch
64 global

Epochs
50

Optimizer
Adam

Loss
MSE
```

This page is important because the project is an academic Gen AI project, not merely a pretty UI.

---

# 22. EXPERIMENT RESULT VISUALIZATION

Provide charts for the Autoencoder experiment:

```text
Training Loss vs Epoch
Validation Loss vs Epoch
```

Also show:

```text
Best Epoch
Best Validation Loss
Final Epoch Loss
Test MSE
```

Do not hardcode future values.

Load metrics from the experiment output when available.

Historical values may be displayed only when explicitly marked as historical.

---

# 23. ABOUT PROJECT PAGE

Explain the academic purpose.

Suggested structure:

```text
GEN AI PROJECT

A multi-model Generative AI pipeline for 2D
game asset generation and processing.

Models

Transformer
Game understanding and structured generation

Autoencoder
Reconstruction and denoising

VAE
Latent-space variation

GAN
Texture/material generation and refinement

Diffusion
Visual generation
```

Also include:

```text
Dataset
Model Architecture
Experiment Results
Responsible AI
```

---

# 24. RESPONSIBLE AI / LICENSE INFORMATION

The project uses multiple external datasets.

The UI should provide a simple transparency section.

Show:

```text
Dataset
License
Purpose
Source
```

Do not claim every dataset is under the same license.

The final dataset ecosystem currently includes sources discussed in the project planning such as:

- OpenGameArt-CC0
- Kenney
- GameTileNet
- VastTextures
- MatSynth
- ViGGO
- NPC-Dialogue v2

The exact final dataset selection and licensing must be verified before presenting it as final.

---

# 25. DEMO MODE

The application must work even when all ML backends are not available.

Create a clearly defined:

```text
DEMO MODE
```

In demo mode:

- UI navigation works
- sample project data can be displayed
- model cards work
- experiment results can be displayed from saved files
- placeholders can demonstrate layout

But every simulated output must be clearly labeled:

```text
Demo Output
```

Never disguise a static sample as a live model result.

---

# 26. BACKEND INTEGRATION RULE

Separate frontend and model services.

Use a clean abstraction such as:

```text
frontend
    ↓
API / service layer
    ↓
model adapters
    ├── transformer
    ├── diffusion
    ├── vae
    ├── autoencoder
    └── gan
```

The UI must not contain model-specific training logic.

The frontend should call APIs/functions such as:

```text
POST /generate/design
POST /generate/image
POST /generate/variations
POST /reconstruct
POST /generate/texture
GET  /experiments
GET  /assets
```

If the real backend does not exist yet, create interfaces/mocks that can later be replaced.

Do not create fake backend intelligence.

---

# 27. RESPONSIVENESS

Desktop is the primary target because the application will likely be demonstrated on a laptop/projector.

Still support:

```text
1280px+
1024px
768px
```

At smaller widths:

- collapse sidebar
- stack cards
- preserve image visibility
- avoid horizontal scrolling where possible

---

# 28. UX RULES

Follow these rules strictly.

### Rule 1
Every screen must have one obvious primary action.

### Rule 2
Never hide important model status.

### Rule 3
Never show meaningless loading animations for fake processing.

### Rule 4
Never fabricate metrics.

### Rule 5
Never claim a model is connected if it is not.

### Rule 6
Do not bury the five-model architecture.

### Rule 7
Do not turn the project into a generic chatbot.

### Rule 8
Do not add unrelated product features.

### Rule 9
Use consistent terminology across all pages.

### Rule 10
The UI must be understandable during a 5-minute project demonstration.

---

# 29. DEMO FLOW

The main presentation flow should be:

## Step 1

Open Dashboard.

Show:

```text
Prompt
↓
Transformer
↓
Diffusion
↓
VAE / AE / GAN
↓
Asset Library
```

## Step 2

Click:

```text
Create New Asset
```

## Step 3

Enter:

```text
Create a dark fantasy RPG knight with a magical sword.
```

## Step 4

Show Transformer-generated structured specification.

## Step 5

Continue to visual generation.

## Step 6

Show generated visual asset.

## Step 7

Show optional processing:

```text
Autoencoder → Reconstruction
VAE → Variations
GAN → Texture refinement
```

## Step 8

Save the result to Asset Library.

## Step 9

Open Experiments.

Show the actual Autoencoder experiment:

```text
282,511 unique images
1,079,108 parameters
Best historical validation MSE ≈ 0.000491
```

Label the result correctly.

## Step 10

Open About Project.

Show all five model families and their roles.

This creates a coherent academic story.

---

# 30. TECHNICAL QUALITY REQUIREMENTS

The generated application must have:

- reusable components
- centralized theme
- consistent spacing
- accessible contrast
- responsive layout
- proper empty states
- proper error states
- loading states only for real operations
- clean routing
- modular model adapters
- no duplicated UI logic
- no hardcoded fake metrics presented as live values

Use realistic sample data only where needed for demo mode.

---

# 31. COMPONENT STRUCTURE

Create reusable components such as:

```text
AppShell
Sidebar
TopBar
PageHeader
PipelineStepper
ModelCard
PromptEditor
GenerationControls
AssetCard
AssetGrid
AssetPreview
ComparisonViewer
MetricCard
MetricChart
ExperimentCard
StatusBadge
ModelStatus
EmptyState
LoadingState
ErrorState
```

Avoid creating huge monolithic components.

---

# 32. DATA STRUCTURES

Use a consistent asset object:

```javascript
{
  id,
  name,
  type,
  source,
  model,
  prompt,
  imageUrl,
  createdAt,
  status,
  metadata
}
```

Use a consistent experiment object:

```javascript
{
  id,
  name,
  model,
  status,
  dataset,
  metrics,
  configuration,
  history
}
```

Use a consistent model object:

```javascript
{
  id,
  name,
  role,
  status,
  input,
  output,
  description
}
```

---

# 33. IMPORTANT AUTOENCODER FACTS TO DISPLAY

These are grounded project facts.

```text
Dataset source:
evilsocket/alucard-sprites

Original:
312,550 images

Exact duplicates removed:
30,039

Unique:
282,511

Train:
254,260

Validation:
14,126

Test:
14,125

Image:
128 × 128 × 4 RGBA

Parameters:
1,079,108

Encoder output:
8 × 8 × 256

Optimizer:
Adam

Learning rate:
0.001

Loss:
MSE

Global batch:
64

Epochs:
50
```

Known historical results:

```text
10K clean test MSE ≈ 0.001251

First 280K run:
Best validation MSE ≈ 0.00049098
Best epoch = 32
```

The first 280K weights were lost after a runtime restart, so this must be labeled as a **historical training result**, not as the current saved final model.

The second 280K run is intended to reproduce the experiment with corrected checkpoint persistence.

---

# 34. DO NOT CHANGE THE ML EXPERIMENT

The GUI project must not modify the validated Autoencoder training configuration.

Do not change:

```text
architecture
dataset split
batch size
learning rate
optimizer
loss
training objective
```

unless the project owner explicitly changes them.

The current experiment is:

```text
AE_280K_full
```

---

# 35. FILE / ARTIFACT AWARENESS

The GUI should be designed to consume saved experiment artifacts.

Expected Autoencoder artifacts:

```text
AE_280K_final.keras
AE_280K_best.keras
AE_280K_encoder.keras
history.json
training_state.json
```

Expected experiment structure:

```text
AE_280K/
├── checkpoints/
├── models/
├── logs/
├── config/
├── evaluation/
└── samples/
```

The frontend should not assume that these files are available until the backend confirms them.

---

# 36. ERROR HANDLING

Use useful errors.

Bad:

```text
Error
```

Good:

```text
Model unavailable

The selected model backend is not connected.
Demo Mode is available for interface exploration.
```

Bad:

```text
Generation failed
```

Good:

```text
Generation failed

The model service did not return a result.
No asset was saved.
```

Never silently substitute fake data after a backend failure.

---

# 37. FINAL ACCEPTANCE CRITERIA

The build is successful only if:

### Visual

- dark professional AI/game-studio aesthetic
- coherent typography
- consistent cards
- polished navigation
- strong image presentation
- responsive desktop layout

### Product

- dashboard explains the project
- Create workflow works
- Transformer stage is understandable
- Diffusion stage is understandable
- VAE stage is understandable
- Autoencoder stage is understandable
- GAN stage is understandable
- Asset Library works
- Experiments page works
- About page works

### Technical

- components are reusable
- model services are abstracted
- backend status is explicit
- demo mode is clearly labeled
- no fake metrics
- no fake connected models
- no unnecessary features

### Academic

The user must be able to explain:

```text
Why five models?
What does each model do?
What data is used?
What does the Autoencoder experiment prove?
What are the quantitative metrics?
How does the full pipeline connect?
```

from the GUI itself.

---

# 38. FINAL PRODUCT MENTAL MODEL

Build the application around this sentence:

> **"A user describes what they want to create, the Transformer turns the idea into a structured game specification, visual models generate and process the assets, and the final results are organized into a reusable game-asset library."**

Everything in the interface should reinforce that story.

Do not build a collection of disconnected model demos.

Build a single **GEN AI Game Asset Studio**.

---

# 39. PRIORITY ORDER

If time is limited, implement in this order:

```text
P0
Application shell
Dashboard
Create workflow
Asset Generator
Asset Library

P1
Transformer page
Diffusion page
Autoencoder page
Experiment results

P2
VAE page
GAN page
Comparison viewer
Pipeline visualization

P3
Advanced animations
Micro-interactions
Extra visual polish
```

Do not spend half the project making a glowing sidebar while the actual workflow is broken. Humanity has suffered enough dashboards.

---

# 40. ANTIGRAVITY IMPLEMENTATION INSTRUCTION

Start by inspecting the existing repository before changing anything.

Determine:

- current frontend framework
- current backend
- current file structure
- current routes
- existing model integrations
- existing assets
- existing environment configuration

Then implement the GUI **inside the existing project structure** where possible.

Do not unnecessarily rewrite the entire repository.

Preserve working ML code.

If an existing component or service already performs the required function, reuse it.

Before claiming completion, verify:

1. application starts successfully
2. routes work
3. no console-breaking errors
4. main workflow works
5. demo mode works
6. model status is accurate
7. experiment metrics render correctly
8. images render correctly
9. responsive layout does not break
10. no placeholder text remains where real project information is available

---

# 41. MOST IMPORTANT CONSTRAINT

Do not optimize the GUI for "more features."

Optimize it for:

```text
CLARITY
+
ACADEMIC CREDIBILITY
+
VISUAL QUALITY
+
DEMOABILITY
```

The evaluator should understand the project's idea within the first 30 seconds.

The evaluator should understand the five-model architecture within 2 minutes.

The evaluator should be able to see a complete prompt-to-asset workflow within 5 minutes.

That is the target.
