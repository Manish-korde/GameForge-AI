import { useState } from 'react';
import { generateDesign, generateImage, runAEInference, runVAEInference, runNoisyAEInference, runVAEInterpolation, fetchRandomHumanSprite, searchSimilarAssets, clusterAssets, detectDuplicateAsset, getAnomalyScore } from '../services/api';

const CreateAsset = () => {
  const [prompt, setPrompt] = useState('');
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [designSpec, setDesignSpec] = useState(null);
  const [generatedAsset, setGeneratedAsset] = useState(null);
  const [aeResult, setAeResult] = useState(null);
  const [isAELoading, setIsAELoading] = useState(false);
  const [noisyAeResult, setNoisyAeResult] = useState(null);
  const [isNoisyAELoading, setIsNoisyAELoading] = useState(false);
  const [noiseScale, setNoiseScale] = useState(0.2);
  const [isDragging, setIsDragging] = useState(false);
  
  const [vaeResult, setVaeResult] = useState(null);
  const [isVAELoading, setIsVAELoading] = useState(false);
  const [vaeScale, setVaeScale] = useState(1.0);
  
  const [spriteB, setSpriteB] = useState(null);
  const [interpolationAlpha, setInterpolationAlpha] = useState(0.5);
  const [interpolatedResult, setInterpolatedResult] = useState(null);
  const [isInterpolating, setIsInterpolating] = useState(false);
  
  // VAE Suite States
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [clusterData, setClusterData] = useState(null);
  const [isClustering, setIsClustering] = useState(false);
  const [duplicateResult, setDuplicateResult] = useState(null);
  const [isCheckingDuplicate, setIsCheckingDuplicate] = useState(false);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [isCheckingAnomaly, setIsCheckingAnomaly] = useState(false);
  
  const [activeView, setActiveView] = useState('original'); // 'original', 'ae', 'vae'

  const handleGenerateDesign = async () => {
    if (!prompt) return;
    setIsLoading(true);
    const res = await generateDesign(prompt);
    setDesignSpec(res.data);
    setIsLoading(false);
    setStep(2);
  };

  const handleGenerateAsset = async () => {
    setIsLoading(true);
    const res = await generateImage(designSpec);
    setGeneratedAsset(res.data);
    setAeResult(null); // Reset AE result on new image
    setNoisyAeResult(null);
    setVaeResult(null);
    setSpriteB(fetchRandomHumanSprite());
    setInterpolatedResult(null);
    setActiveView('original');
    setIsLoading(false);
    setStep(3);
  };

  const handleRunAE = async () => {
    if (!generatedAsset) return;
    setIsAELoading(true);
    setActiveView('ae');
    try {
      const result = await runAEInference(generatedAsset.imageUrl);
      setAeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run AE Inference");
      setActiveView('original');
    } finally {
      setIsAELoading(false);
    }
  };

  const handleRunNoisyAE = async () => {
    if (!generatedAsset) return;
    setIsNoisyAELoading(true);
    try {
      const result = await runNoisyAEInference(generatedAsset.imageUrl, noiseScale);
      setNoisyAeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run Noisy AE Inference");
    } finally {
      setIsNoisyAELoading(false);
    }
  };

  const handleRunVAE = async () => {
    if (!generatedAsset) return;
    setIsVAELoading(true);
    try {
      const result = await runVAEInference(generatedAsset.imageUrl, vaeScale);
      setVaeResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run VAE Inference");
    } finally {
      setIsVAELoading(false);
    }
  };

  const handleRunInterpolation = async () => {
    if (!generatedAsset || !spriteB) return;
    setIsInterpolating(true);
    try {
      const result = await runVAEInterpolation(generatedAsset.imageUrl, spriteB, interpolationAlpha);
      setInterpolatedResult(result);
    } catch (err) {
      console.error(err);
      alert("Failed to run VAE Interpolation");
    } finally {
      setIsInterpolating(false);
    }
  };

  const handleRandomizeSpriteB = () => {
    setSpriteB(fetchRandomHumanSprite());
    setInterpolatedResult(null);
  };

  // VAE Suite Handlers
  const handleSearchSimilar = async () => {
    if (!generatedAsset) return;
    setIsSearching(true);
    try {
      const res = await searchSimilarAssets(generatedAsset.imageUrl, 4);
      setSearchResults(res);
    } catch (err) {
      console.error(err);
      alert("Failed visual search");
    } finally {
      setIsSearching(false);
    }
  };

  const handleClusterAssets = async () => {
    if (!generatedAsset || !spriteB) return;
    setIsClustering(true);
    try {
      const pool = [
        generatedAsset.imageUrl,
        spriteB,
        "http://localhost:8000/alucard_samples/alucard_2.png",
        "http://localhost:8000/alucard_samples/alucard_5.png",
        "http://localhost:8000/alucard_samples/alucard_8.png",
        "http://localhost:8000/alucard_samples/alucard_12.png"
      ];
      const res = await clusterAssets(pool);
      setClusterData(res);
    } catch (err) {
      console.error(err);
      alert("Failed clustering");
    } finally {
      setIsClustering(false);
    }
  };

  const handleDetectDuplicate = async () => {
    if (!generatedAsset || !spriteB) return;
    setIsCheckingDuplicate(true);
    try {
      const res = await detectDuplicateAsset(generatedAsset.imageUrl, spriteB);
      setDuplicateResult(res);
    } catch (err) {
      console.error(err);
      alert("Failed duplicate check");
    } finally {
      setIsCheckingDuplicate(false);
    }
  };

  const handleCheckAnomaly = async () => {
    if (!generatedAsset) return;
    setIsCheckingAnomaly(true);
    try {
      const res = await getAnomalyScore(generatedAsset.imageUrl);
      setAnomalyResult(res);
    } catch (err) {
      console.error(err);
      alert("Failed anomaly detection");
    } finally {
      setIsCheckingAnomaly(false);
    }
  };


  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFileUpload(file);
    }
  };
  
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };
  
  const handleFileUpload = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }
    const localUrl = URL.createObjectURL(file);
    setGeneratedAsset({
      id: 'custom-' + Date.now(),
      name: file.name,
      type: 'Image',
      source: 'Upload',
      model: 'Custom Upload',
      prompt: 'Custom user uploaded image for AE testing',
      imageUrl: localUrl,
      createdAt: new Date().toISOString(),
      status: 'Completed'
    });
    setAeResult(null);
    setNoisyAeResult(null);
    setVaeResult(null);
    setSpriteB(fetchRandomHumanSprite());
    setInterpolatedResult(null);
    setActiveView('original');
    setStep(3);
  };

  return (
    <div className="create-asset">
      <h2>Create Game Asset</h2>

      {step === 1 && (
        <div className="card mt-4">
          <label className="block mb-2 text-sm">Prompt</label>
          <textarea 
            className="input textarea mb-4" 
            placeholder="Describe the game asset you want to create..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          ></textarea>
          
          <div className="flex gap-sm mb-6">
            <button className="chip" onClick={() => setPrompt("Dark fantasy knight")}>Dark fantasy knight</button>
            <button className="chip" onClick={() => setPrompt("Pixel-art forest village")}>Pixel-art forest village</button>
            <button className="chip" onClick={() => setPrompt("Magic sword")}>Magic sword</button>
          </div>

          <div className="grid grid-cols-3 gap-md mb-6">
            <div>
              <label className="block text-sm mb-1">Asset Type</label>
              <select className="input"><option>Character</option></select>
            </div>
            <div>
              <label className="block text-sm mb-1">Style</label>
              <select className="input"><option>Pixel Art</option></select>
            </div>
            <div>
              <label className="block text-sm mb-1">Generation Mode</label>
              <select className="input"><option>Full Pipeline</option></select>
            </div>
          </div>

          <button className="btn btn-primary" onClick={handleGenerateDesign} disabled={isLoading}>
            {isLoading ? 'Generating...' : 'Generate Design'}
          </button>
        </div>
      )}

      {step === 2 && designSpec && (
        <div className="grid grid-cols-2 gap-lg mt-4">
          <div className="card">
            <h3 className="mb-4">GAME CONCEPT</h3>
            <div className="grid grid-cols-2 gap-md mb-4">
              <div>
                <span className="text-sm text-secondary block">Theme</span>
                {designSpec.theme}
              </div>
              <div>
                <span className="text-sm text-secondary block">Environment</span>
                {designSpec.environment}
              </div>
              <div>
                <span className="text-sm text-secondary block">Characters</span>
                {designSpec.characters.length}
              </div>
              <div>
                <span className="text-sm text-secondary block">Visual Style</span>
                {designSpec.visualStyle}
              </div>
            </div>
            <div className="flex gap-md mt-6">
              <button className="btn btn-secondary" onClick={() => setStep(1)}>Edit Specification</button>
              <button className="btn btn-primary" onClick={handleGenerateAsset} disabled={isLoading}>
                {isLoading ? 'Generating Image...' : 'Continue to Visual Generation'}
              </button>
            </div>
          </div>
          <div className="card">
             <h3 className="mb-4 text-sm uppercase text-secondary">Transformer Details</h3>
             <pre className="raw-output">{designSpec.rawOutput}</pre>
          </div>
        </div>
      )}

      {step === 3 && generatedAsset && (
        <div className="grid grid-cols-3 gap-lg mt-4">
          <div className="col-span-1">
             <div className="card">
               <h3 className="mb-4">Asset Details</h3>
               <div className="text-sm text-secondary mb-1">Name</div>
               <div className="mb-4">{generatedAsset.name}</div>
               <div className="text-sm text-secondary mb-1">Prompt</div>
               <div className="mb-4 text-sm">{generatedAsset.prompt}</div>
               <button className="btn btn-secondary w-full mb-2" onClick={handleRunAE} disabled={isAELoading || isVAELoading}>
                 {isAELoading ? 'Running AE...' : 'Run through AE'}
               </button>
               <button className="btn btn-secondary w-full mb-2" onClick={() => setActiveView('vae')} disabled={isAELoading || isVAELoading}>
                 Create VAE Variations
               </button>
             </div>
          </div>
          <div className="col-span-2">
            <div className="card text-center flex flex-col items-center h-full">
              {activeView === 'original' && (
                <div 
                  className={`w-full h-full min-h-[300px] flex flex-col items-center justify-center border-2 border-dashed rounded-lg transition-colors relative ${isDragging ? 'border-primary bg-primary/10' : 'border-[#333] hover:border-gray-500'}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <img src={generatedAsset.imageUrl} alt="Generated" className="generated-preview mb-4 object-contain max-h-[300px]" />
                  <div className="flex gap-md mb-2">
                    <button className="btn btn-primary">Save to Library</button>
                  </div>
                  
                  {/* Drag and Drop Overlay / Controls */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center opacity-0 hover:opacity-100 bg-black/60 rounded-lg transition-opacity">
                    <p className="mb-2 font-medium">Drag & Drop to swap image</p>
                    <p className="text-sm text-secondary mb-4">or</p>
                    <label className="btn btn-secondary cursor-pointer">
                      Browse Files
                      <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                    </label>
                  </div>
                </div>
              )}

              {activeView === 'ae' && !aeResult && (
                <div className="flex-1 flex items-center justify-center">Loading AE Results...</div>
              )}

              {activeView === 'ae' && aeResult && (
                <div className="w-full">
                  <h4 className="mb-6 text-xl font-bold">Autoencoder Inference</h4>
                  <div className="flex gap-lg justify-center mb-8">
                    <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                      <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Original (Scaled)</span>
                      <img src={aeResult.original_processed} alt="Original processed" className="rounded shadow-sm" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                    </div>
                    <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                      <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Reconstructed</span>
                      <img src={aeResult.reconstructed} alt="Reconstructed" className="rounded shadow-sm" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                    </div>
                  </div>

                  {aeResult.metrics && (
                    <div className="bg-[#151515] rounded-xl p-6 border border-border mb-8 max-w-2xl mx-auto shadow-lg">
                      <h5 className="text-sm text-secondary uppercase tracking-wider mb-4 font-semibold text-left">Real-Time Evaluation Metrics</h5>
                      
                      {/* Primary Error Metrics */}
                      <div className="grid grid-cols-2 gap-4 text-center mb-4">
                        <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                          <div className="text-xs text-secondary mb-1">MSE Loss</div>
                          <div className="font-mono text-primary text-xl font-bold">{aeResult.metrics.mse.toFixed(6)}</div>
                        </div>
                        <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                          <div className="text-xs text-secondary mb-1">MAE Loss</div>
                          <div className="font-mono text-primary text-xl font-bold">{aeResult.metrics.mae.toFixed(6)}</div>
                        </div>
                      </div>

                      {/* Visual Accuracy Metrics */}
                      <div className="grid grid-cols-3 gap-4 text-center mb-6">
                        <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                          <div className="text-xs text-secondary mb-1">Color Palette Match</div>
                          <div className="font-mono text-green-400 font-bold">{aeResult.metrics.color_match.toFixed(2)}%</div>
                        </div>
                        <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                          <div className="text-xs text-secondary mb-1">Exact Pixel Match</div>
                          <div className="font-mono text-green-400 font-bold">{aeResult.metrics.exact_match.toFixed(2)}%</div>
                        </div>
                        <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                          <div className="text-xs text-secondary mb-1">Alpha Mask IoU</div>
                          <div className="font-mono text-green-400 font-bold">{aeResult.metrics.alpha_iou.toFixed(2)}%</div>
                        </div>
                      </div>

                      {/* Silent Footer Metrics */}
                      <div className="flex justify-between items-center text-[#555] text-[10px] border-t border-[#333] pt-3 px-2">
                        <span>PSNR: {aeResult.metrics.psnr.toFixed(2)} dB</span>
                        <span>SSIM: {aeResult.metrics.ssim.toFixed(4)}</span>
                      </div>
                    </div>
                  )}

                  {/* Noisy AE Section */}
                  <div className="mt-12 border-t border-[#333] pt-8">
                    <h4 className="mb-6 text-xl font-bold">Denoising Autoencoder Test</h4>
                    <div className="mb-6 max-w-md mx-auto">
                      <label className="block text-sm text-secondary mb-2">Noise Level: {noiseScale}</label>
                      <input 
                        type="range" 
                        min="0.0" 
                        max="1.0" 
                        step="0.05" 
                        value={noiseScale} 
                        onChange={(e) => setNoiseScale(parseFloat(e.target.value))}
                        className="w-full mb-4 accent-primary"
                      />
                      <button className="btn btn-primary w-full" onClick={handleRunNoisyAE} disabled={isNoisyAELoading}>
                        {isNoisyAELoading ? 'Adding Noise & Reconstructing...' : 'Add Noise & Reconstruct'}
                      </button>
                    </div>

                    {noisyAeResult && (
                      <>
                        <div className="flex gap-lg justify-center mb-8">
                          <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                            <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Original</span>
                            <img src={noisyAeResult.original_processed} alt="Original" className="rounded shadow-sm" style={{ width: '200px', height: '200px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                          </div>
                          <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                            <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Noisy Input</span>
                            <img src={noisyAeResult.noisy_image} alt="Noisy Input" className="rounded shadow-sm" style={{ width: '200px', height: '200px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                          </div>
                          <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                            <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Reconstructed</span>
                            <img src={noisyAeResult.reconstructed} alt="Reconstructed" className="rounded shadow-sm" style={{ width: '200px', height: '200px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                          </div>
                        </div>

                        <div className="bg-[#151515] rounded-xl p-6 border border-border mb-8 max-w-2xl mx-auto shadow-lg">
                          <h5 className="text-sm text-secondary uppercase tracking-wider mb-4 font-semibold text-left">Denoising Evaluation Metrics</h5>
                          <div className="grid grid-cols-2 gap-4 text-center mb-4">
                            <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">MSE Loss</div>
                              <div className="font-mono text-primary text-xl font-bold">{noisyAeResult.metrics.mse.toFixed(6)}</div>
                            </div>
                            <div className="bg-[#1a1a1a] p-4 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">MAE Loss</div>
                              <div className="font-mono text-primary text-xl font-bold">{noisyAeResult.metrics.mae.toFixed(6)}</div>
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-center mb-6">
                            <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">Color Palette Match</div>
                              <div className="font-mono text-green-400 font-bold">{noisyAeResult.metrics.color_match.toFixed(2)}%</div>
                            </div>
                            <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">Exact Pixel Match</div>
                              <div className="font-mono text-green-400 font-bold">{noisyAeResult.metrics.exact_match.toFixed(2)}%</div>
                            </div>
                            <div className="bg-[#1a1a1a] p-3 rounded-lg border border-[#333]">
                              <div className="text-xs text-secondary mb-1">Alpha Mask IoU</div>
                              <div className="font-mono text-green-400 font-bold">{noisyAeResult.metrics.alpha_iou.toFixed(2)}%</div>
                            </div>
                          </div>
                          <div className="flex justify-between items-center text-[#555] text-[10px] border-t border-[#333] pt-3 px-2">
                            <span>PSNR: {noisyAeResult.metrics.psnr.toFixed(2)} dB</span>
                            <span>SSIM: {noisyAeResult.metrics.ssim.toFixed(4)}</span>
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="flex gap-md justify-center">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setActiveView('original')}>Back to Original</button>
                  </div>
                </div>
              )}

              {activeView === 'vae' && (
                <div className="w-full">
                  {/* Top 5 VAE Applications Panel */}
                  <div className="mb-12 bg-[#121215] border border-primary/40 rounded-2xl p-6 shadow-2xl">
                    <div className="flex items-center justify-between mb-6 border-b border-[#25252e] pb-4">
                      <div>
                        <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                          <span className="text-primary">🔥</span> VAE Core Applications Suite
                        </h3>
                        <p className="text-xs text-secondary mt-1">High-value latent space features tailored for GameForge AI</p>
                      </div>
                      <span className="bg-primary/20 text-primary border border-primary/40 text-xs px-3 py-1 rounded-full font-mono">280k VAE Latent Engine</span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                      {/* App 1: Visual Search */}
                      <div className="bg-[#1a1a20] rounded-xl p-4 border border-[#2a2a35] hover:border-primary/50 transition">
                        <div className="text-xs font-semibold text-primary uppercase tracking-wider mb-2">1. Visual Search</div>
                        <p className="text-xs text-secondary mb-4">Find top-K visually similar assets in latent space</p>
                        <button className="btn btn-primary w-full text-xs py-2" onClick={handleSearchSimilar} disabled={isSearching}>
                          {isSearching ? 'Searching...' : 'Find Similar Assets'}
                        </button>
                      </div>

                      {/* App 2: Asset Clustering */}
                      <div className="bg-[#1a1a20] rounded-xl p-4 border border-[#2a2a35] hover:border-primary/50 transition">
                        <div className="text-xs font-semibold text-primary uppercase tracking-wider mb-2">2. Asset Clustering</div>
                        <p className="text-xs text-secondary mb-4">Project assets to 2D space & auto-group clusters</p>
                        <button className="btn btn-primary w-full text-xs py-2" onClick={handleClusterAssets} disabled={isClustering}>
                          {isClustering ? 'Clustering...' : 'Cluster Library Assets'}
                        </button>
                      </div>

                      {/* App 3: Duplicate Detection */}
                      <div className="bg-[#1a1a20] rounded-xl p-4 border border-[#2a2a35] hover:border-primary/50 transition">
                        <div className="text-xs font-semibold text-primary uppercase tracking-wider mb-2">3. Duplicate Detection</div>
                        <p className="text-xs text-secondary mb-4">Check latent distance vs Character B</p>
                        <button className="btn btn-primary w-full text-xs py-2" onClick={handleDetectDuplicate} disabled={isCheckingDuplicate}>
                          {isCheckingDuplicate ? 'Checking...' : 'Detect Duplicates'}
                        </button>
                      </div>

                      {/* App 4: Anomaly Detection */}
                      <div className="bg-[#1a1a20] rounded-xl p-4 border border-[#2a2a35] hover:border-primary/50 transition">
                        <div className="text-xs font-semibold text-primary uppercase tracking-wider mb-2">4. Anomaly Detection</div>
                        <p className="text-xs text-secondary mb-4">Score asset quality & out-of-domain status</p>
                        <button className="btn btn-primary w-full text-xs py-2" onClick={handleCheckAnomaly} disabled={isCheckingAnomaly}>
                          {isCheckingAnomaly ? 'Evaluating...' : 'Check Anomaly Score'}
                        </button>
                      </div>
                    </div>

                    {/* Results Panels */}
                    {/* Visual Search Results */}
                    {searchResults && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] mb-6">
                        <h4 className="text-sm font-semibold text-white mb-3">Top Visually Similar Assets (Latent Nearest Neighbors)</h4>
                        <div className="flex gap-4 overflow-x-auto pb-2">
                          {searchResults.matches.map((item, idx) => (
                            <div key={idx} className="flex flex-col items-center bg-[#101014] p-3 rounded-lg border border-[#2a2a35] min-w-[110px]">
                              <img src={item.url} alt="Match" className="w-16 h-16 object-contain mb-2" style={{ imageRendering: 'pixelated' }} />
                              <span className="text-[11px] text-green-400 font-mono font-bold">{item.similarity}% Match</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Clustering Results */}
                    {clusterData && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] mb-6">
                        <h4 className="text-sm font-semibold text-white mb-3">2D Latent Space Map & Cluster Organization</h4>
                        <div className="grid grid-cols-3 gap-3">
                          {clusterData.clusters.map((item, idx) => (
                            <div key={idx} className="flex items-center gap-3 bg-[#101014] p-2 rounded-lg border border-[#2a2a35]">
                              <img src={item.url} alt="Cluster Item" className="w-10 h-10 object-contain" style={{ imageRendering: 'pixelated' }} />
                              <div className="text-left">
                                <div className="text-[10px] text-primary font-bold">Cluster #{item.cluster}</div>
                                <div className="text-[9px] text-secondary font-mono">({item.x}, {item.y})</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Duplicate Check Results */}
                    {duplicateResult && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] mb-6 flex justify-between items-center">
                        <div className="text-left">
                          <h4 className="text-sm font-semibold text-white">Duplicate Detection Report</h4>
                          <p className="text-xs text-secondary mt-1">Latent Distance: <span className="font-mono text-white">{duplicateResult.latent_distance}</span> | Pixel MSE: <span className="font-mono text-white">{duplicateResult.pixel_mse}</span></p>
                        </div>
                        <div className={`px-4 py-2 rounded-lg font-bold text-xs uppercase font-mono ${duplicateResult.is_duplicate ? 'bg-red-900/60 text-red-300 border border-red-500' : 'bg-green-900/60 text-green-300 border border-green-500'}`}>
                          {duplicateResult.status} ({duplicateResult.similarity_score}%)
                        </div>
                      </div>
                    )}

                    {/* Anomaly Check Results */}
                    {anomalyResult && (
                      <div className="bg-[#18181f] p-4 rounded-xl border border-[#333] mb-6 text-left">
                        <div className="flex justify-between items-center mb-2">
                          <h4 className="text-sm font-semibold text-white">VAE Quality & Anomaly Diagnostic</h4>
                          <span className="text-xs font-mono text-primary font-bold">Score: {anomalyResult.anomaly_score}</span>
                        </div>
                        <div className="grid grid-cols-3 gap-3 text-center mb-3">
                          <div className="bg-[#101014] p-2 rounded border border-[#252530]">
                            <div className="text-[10px] text-secondary">Recon MSE</div>
                            <div className="text-xs font-mono font-bold text-white">{anomalyResult.reconstruction_mse}</div>
                          </div>
                          <div className="bg-[#101014] p-2 rounded border border-[#252530]">
                            <div className="text-[10px] text-secondary">KL Divergence</div>
                            <div className="text-xs font-mono font-bold text-white">{anomalyResult.kl_divergence}</div>
                          </div>
                          <div className="bg-[#101014] p-2 rounded border border-[#252530]">
                            <div className="text-[10px] text-secondary">Classification</div>
                            <div className="text-xs font-bold text-green-400">{anomalyResult.classification}</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Interpolation Section at the Top */}
                  <div className="mb-12">
                    <h4 className="mb-6 text-xl font-bold">VAE Character Interpolation</h4>
                    
                    <div className="flex gap-lg justify-center mb-6">
                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Character A</span>
                        <img src={generatedAsset.imageUrl} alt="Sprite A" className="rounded shadow-sm" style={{ width: '128px', height: '128px', imageRendering: 'pixelated', border: '1px solid #333', objectFit: 'contain' }} />
                      </div>
                      
                      <div className="flex flex-col items-center justify-center">
                        <div className="w-[128px] flex flex-col items-center">
                          <label className="text-xs text-secondary mb-1">Alpha: {interpolationAlpha.toFixed(2)}</label>
                          <input 
                            type="range" 
                            min="0.0" 
                            max="1.0" 
                            step="0.05" 
                            value={interpolationAlpha} 
                            onChange={(e) => setInterpolationAlpha(parseFloat(e.target.value))}
                            className="w-full mb-2 accent-primary"
                          />
                          <button className="btn btn-primary w-full text-xs py-1" onClick={handleRunInterpolation} disabled={isInterpolating}>
                            {isInterpolating ? 'Merging...' : 'Interpolate'}
                          </button>
                        </div>
                      </div>

                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-sm text-secondary uppercase tracking-wider font-semibold">Character B</span>
                          <button className="text-xs bg-[#333] hover:bg-[#444] px-2 py-0.5 rounded text-white" onClick={handleRandomizeSpriteB}>Swap</button>
                        </div>
                        <img src={spriteB} alt="Sprite B" className="rounded shadow-sm" style={{ width: '128px', height: '128px', imageRendering: 'pixelated', border: '1px solid #333', objectFit: 'contain' }} />
                      </div>
                    </div>

                    <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-primary shadow-lg max-w-sm mx-auto">
                      <span className="text-sm text-primary mb-3 uppercase tracking-wider font-semibold">Interpolated Result</span>
                      {interpolatedResult ? (
                        <img src={interpolatedResult.interpolated} alt="Interpolated" className="rounded shadow-sm bg-black" style={{ width: '192px', height: '192px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                      ) : (
                        <div className="w-[192px] h-[192px] border border-[#333] rounded flex items-center justify-center bg-[#1a1a1a] text-secondary text-sm text-center p-4">
                          {isInterpolating ? 'Interpolating in latent space...' : 'Click Interpolate to merge characters'}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Variation Section at the Bottom */}
                  <div className="border-t border-[#333] pt-8">
                    <h4 className="mb-6 text-xl font-bold">VAE Asset Variation</h4>
                    
                    <div className="mb-6 max-w-md mx-auto">
                      <label className="block text-sm text-secondary mb-2">Variation Scale: {vaeScale}</label>
                      <input 
                        type="range" 
                        min="0.0" 
                        max="1.5" 
                        step="0.1" 
                        value={vaeScale} 
                        onChange={(e) => setVaeScale(parseFloat(e.target.value))}
                        className="w-full mb-4 accent-primary"
                      />
                      <button className="btn btn-primary w-full" onClick={handleRunVAE} disabled={isVAELoading}>
                        {isVAELoading ? 'Generating Variant...' : 'Generate Variation'}
                      </button>
                    </div>

                    <div className="flex gap-lg justify-center mb-8">
                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Original</span>
                        <img src={generatedAsset.imageUrl} alt="Original" className="rounded shadow-sm" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333', objectFit: 'contain' }} />
                      </div>
                      <div className="flex flex-col items-center bg-[#151515] rounded-xl p-4 border border-border shadow-lg">
                        <span className="text-sm text-secondary mb-3 uppercase tracking-wider font-semibold">Variant</span>
                        {vaeResult ? (
                          <img src={vaeResult.variant} alt="Variant" className="rounded shadow-sm bg-black" style={{ width: '256px', height: '256px', imageRendering: 'pixelated', border: '1px solid #333' }} />
                        ) : (
                          <div className="w-[256px] h-[256px] border border-[#333] rounded flex items-center justify-center bg-[#1a1a1a] text-secondary text-sm">
                            {isVAELoading ? 'Generating...' : 'No variant generated yet'}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-md justify-center mt-8">
                    <button className="btn btn-secondary px-8 py-2" onClick={() => setActiveView('original')}>Back to Original</button>
                  </div>
                </div>
              )}

            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CreateAsset;
