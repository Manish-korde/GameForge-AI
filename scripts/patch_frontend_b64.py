import os
from pathlib import Path

target_file = Path("gui/src/pages/ActionStudio.jsx")
with open(target_file, "r", encoding="utf-8") as f:
    code = f.read()

target = '''  // Single Pose Generation
  const handleGenerateSingle = async () => {
    if (!activeRefDisplayUrl) {
      setError("Please select or upload a character reference sprite.");
      return;
    }'''

replacement = '''  // Helper to ensure base64
  const ensureBase64 = async (urlOrB64) => {
    if (!urlOrB64) return null;
    if (urlOrB64.startsWith("data:image")) return urlOrB64;
    try {
      const resp = await fetch(urlOrB64);
      const blob = await resp.blob();
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      return urlOrB64;
    }
  };

  // Single Pose Generation
  const handleGenerateSingle = async () => {
    if (!activeRefDisplayUrl) {
      setError("Please select or upload a character reference sprite.");
      return;
    }'''

assert target in code, "Target not found in ActionStudio.jsx"
code = code.replace(target, replacement, 1)

code = code.replace(
    'referenceB64: activeRefDisplayUrl,',
    'referenceB64: await ensureBase64(activeRefDisplayUrl),',
    2
)

with open(target_file, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated ActionStudio.jsx with client-side ensureBase64!")

