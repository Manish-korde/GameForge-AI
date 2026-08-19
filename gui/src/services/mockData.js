export const mockExperiments = [
  {
    id: 'exp-ae-280k',
    name: 'Autoencoder',
    model: 'Autoencoder',
    status: 'Running',
    dataset: {
      source: 'evilsocket/alucard-sprites',
      uniqueImages: 282511,
      train: 254260,
      validation: 14126,
      test: 14125
    },
    metrics: {
      bestValidationMse: 0.00049098,
      bestEpoch: 32,
      tenKBaselineMse: 0.001251
    },
    configuration: {
      imageShape: '128 × 128 × 4 RGBA',
      parameters: 1079108,
      encoderOutput: '8 × 8 × 256',
      optimizer: 'Adam',
      learningRate: 0.001,
      loss: 'MSE',
      globalBatch: 64,
      epochs: 50
    }
  },
  {
    id: 'exp-vae',
    name: 'VAE',
    model: 'Variational Autoencoder',
    status: 'Demo / Not Connected'
  },
  {
    id: 'exp-gan',
    name: 'GAN Texture Studio',
    model: 'GAN',
    status: 'Demo / Not Connected'
  },
  {
    id: 'exp-diffusion',
    name: 'Primary Visual Generation',
    model: 'Diffusion',
    status: 'Demo'
  },
  {
    id: 'exp-transformer',
    name: 'Game Concept Spec',
    model: 'Transformer',
    status: 'Demo'
  }
];

export const mockStructuredDesign = (userPrompt) => ({
  theme: 'Dark Fantasy',
  environment: 'Cursed Forest Village',
  characters: [
    { name: 'Village Elder', desc: 'Old wise figure in torn robes' },
    { name: 'Guard', desc: 'Weary soldier with rusted armor' }
  ],
  weapons: [
    { name: 'Magical Sword', desc: 'Glowing blue rune blade' }
  ],
  props: [
    { name: 'Wooden Sign', desc: 'Broken signpost covered in moss' }
  ],
  visualStyle: 'Pixel Art',
  rawOutput: `Theme:\nDark fantasy\n\nEnvironment:\nCursed forest village\n\nCharacters:\n- Village elder\n- Guard\n- Witch\n\nAssets:\n- Wooden houses\n- Broken sign\n- Lanterns\n- Fog\n- Magic symbols\n\nOriginal prompt: ${userPrompt}`
});

export const mockAssets = [
  {
    id: 'asset-1',
    name: 'Cursed Knight',
    type: 'Character',
    source: 'Diffusion',
    model: 'Diffusion Model',
    prompt: 'Pixel-art knight with a magical sword in dark fantasy style',
    imageUrl: 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/150.png',
    createdAt: new Date(Date.now() - 100000).toISOString(),
    status: 'Completed'
  },
  {
    id: 'asset-2',
    name: 'Mossy Stone',
    type: 'Texture',
    source: 'GAN',
    model: 'GAN',
    prompt: 'Mossy dungeon stone texture',
    imageUrl: 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/95.png',
    createdAt: new Date(Date.now() - 200000).toISOString(),
    status: 'Completed'
  }
];
