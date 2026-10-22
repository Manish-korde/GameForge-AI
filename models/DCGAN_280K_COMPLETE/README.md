# DCGAN 280K Artifact Bundle

This archive contains DCGAN model files, configuration, logs, samples, and resumable TensorFlow checkpoints when training has completed at least one epoch.

## Resume in a fresh Kaggle session
1. Attach this ZIP as a Kaggle input dataset.
2. Set RESUME_ARCHIVE_PATH to the mounted ZIP path in Section 3.
3. Set TARGET_EPOCHS=100 (total target) and RESUME_MODE='auto' or 'required'.
4. Run cells in order. The archive is extracted only when no local checkpoint exists.

Completed epochs at export: 50.

## Important
Kaggle /kaggle/working storage is temporary. Download the ZIP or save a versioned Kaggle output before the session ends.
