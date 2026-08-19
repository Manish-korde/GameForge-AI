import { useState, useEffect } from 'react';
import { fetchExperiments } from '../services/api';

const Experiments = () => {
  const [experiments, setExperiments] = useState([]);

  useEffect(() => {
    fetchExperiments().then(res => setExperiments(res.data));
  }, []);

  return (
    <div className="experiments-page">
      <h2>Experiments & Models</h2>
      <p className="text-secondary mb-6">Academic status and historical metrics of the Generative AI models.</p>

      <div className="grid grid-cols-2 gap-lg">
        {experiments.map(exp => (
          <div key={exp.id} className="card">
            <div className="flex justify-between items-center mb-4">
              <h3 className="m-0">{exp.name}</h3>
              <span className={`status-badge ${exp.status.includes('Running') ? 'active' : ''}`}>{exp.status}</span>
            </div>
            <div className="text-sm text-secondary mb-4">Model: {exp.model}</div>
            
            {exp.dataset && (
              <div className="experiment-details">
                <div className="grid grid-cols-2 gap-md text-sm mb-4">
                  <div>
                    <strong className="block text-secondary">Dataset</strong>
                    {exp.dataset.uniqueImages.toLocaleString()} unique images<br/>
                    (Train: {exp.dataset.train.toLocaleString()}, Val: {exp.dataset.validation.toLocaleString()})
                  </div>
                  <div>
                    <strong className="block text-secondary">Architecture</strong>
                    {exp.configuration.parameters.toLocaleString()} parameters<br/>
                    {exp.configuration.encoderOutput} encoder output
                  </div>
                </div>

                <div className="metrics-box p-4 bg-surface-elevated rounded border border-border mt-4">
                  <h4 className="text-sm text-secondary uppercase mb-2">Historical Training Result</h4>
                  <div className="text-2xl text-primary mb-1">{exp.metrics.bestValidationMse.toFixed(8)}</div>
                  <div className="text-sm text-secondary">Best Validation MSE (Epoch {exp.metrics.bestEpoch})</div>
                  <div className="text-xs text-secondary mt-2">10K Baseline MSE: {exp.metrics.tenKBaselineMse}</div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Experiments;
