path = 'gui/src/pages/ActionStudio.jsx'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add state variable
t1 = '  const [seed, setSeed] = useState("42");'
r1 = '  const [seed, setSeed] = useState("42");\n  const [enableF7Patch, setEnableF7Patch] = useState(false);'

if t1 not in content:
    t1 = t1.replace('\n', '\r\n')
    r1 = r1.replace('\n', '\r\n')

assert t1 in content, "t1 not found"
content = content.replace(t1, r1, 1)

# 2. Update handleGenerateSingle
t2 = """      const res = await generateDiffusionPose({
        action,
        poseId: action === "idle" ? 0 : poseId,
        referenceB64: await ensureBase64(activeRefDisplayUrl),
        steps: parseInt(steps) || 50,
        seed: seed ? parseInt(seed) : null
      });"""

r2 = """      const res = await generateDiffusionPose({
        action,
        poseId: action === "idle" ? 0 : poseId,
        referenceB64: await ensureBase64(activeRefDisplayUrl),
        steps: parseInt(steps) || 50,
        seed: seed ? parseInt(seed) : null,
        enableF7Patch
      });"""

if t2 not in content:
    t2 = t2.replace('\n', '\r\n')
    r2 = r2.replace('\n', '\r\n')

assert t2 in content, "t2 not found"
content = content.replace(t2, r2, 1)

# 3. Update handleGenerateWalkCycle
t3 = """      const res = await generateDiffusionWalkCycle({
        referenceB64: await ensureBase64(activeRefDisplayUrl),
        steps: parseInt(steps) || 50,
        seed: seed ? parseInt(seed) : 42
      });"""

r3 = """      const res = await generateDiffusionWalkCycle({
        referenceB64: await ensureBase64(activeRefDisplayUrl),
        steps: parseInt(steps) || 50,
        seed: seed ? parseInt(seed) : 42,
        enableF7Patch
      });"""

if t3 not in content:
    t3 = t3.replace('\n', '\r\n')
    r3 = r3.replace('\n', '\r\n')

assert t3 in content, "t3 not found"
content = content.replace(t3, r3, 1)

# 4. Insert Toggle UI right above Action Buttons
toggle_ui = """            {/* Experimental Bilateral Stride Mirror Patch Toggle */}
            <div style={{
              background: enableF7Patch ? "rgba(16, 185, 129, 0.15)" : "rgba(15, 23, 42, 0.6)",
              border: enableF7Patch ? "1px solid rgba(16, 185, 129, 0.4)" : "1px solid rgba(148, 163, 184, 0.2)",
              borderRadius: 8,
              padding: "0.65rem 0.85rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "0.85rem",
              transition: "all 0.15s ease"
            }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: enableF7Patch ? "#34d399" : "#cbd5e1" }}>
                  Bilateral F7 Stride Mirror Patch (Beta)
                </div>
                <div style={{ fontSize: 10, color: "#94a3b8" }}>
                  Feathered mirror from F3 left-apex to address 38.9% right-stride compression.
                </div>
              </div>
              <input
                type="checkbox"
                checked={enableF7Patch}
                onChange={(e) => setEnableF7Patch(e.target.checked)}
                style={{ width: 16, height: 16, accentColor: "#10b981", cursor: "pointer" }}
              />
            </div>

            {/* Action Buttons */}"""

t4 = "            {/* Action Buttons */}"
assert t4 in content, "t4 not found"
content = content.replace(t4, toggle_ui, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("ActionStudio.jsx updated successfully with enableF7Patch toggle!")
