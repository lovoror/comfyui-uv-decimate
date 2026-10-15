import { app } from "../../scripts/app.js";

const NODE_NAME = "DecimateMeshKeepUV";

const PRESET_TO_FACES = new Map([
    ["1k", 1000],
    ["2k", 2000],
    ["5k", 5000],
    ["10k", 10000],
    ["20k", 20000],
    ["30k", 30000],
    ["50k (recommended)", 50000],
    ["50k", 50000],
    ["100k", 100000],
]);

const FACES_TO_PRESET = new Map([
    [1000, "1k"],
    [2000, "2k"],
    [5000, "5k"],
    [10000, "10k"],
    [20000, "20k"],
    [30000, "30k"],
    [50000, "50k (recommended)"],
    [100000, "100k"],
]);

function bindPresetAndTargetFaces(node) {
    const presetWidget = node.widgets?.find((w) => w.name === "preset");
    const targetFacesWidget = node.widgets?.find((w) => w.name === "target_faces");
    if (!presetWidget || !targetFacesWidget) {
        return;
    }

    if (node._uvDecimatePresetBound) {
        const mapped = PRESET_TO_FACES.get(String(presetWidget.value));
        if (mapped !== undefined && targetFacesWidget.value !== mapped) {
            targetFacesWidget.value = mapped;
        }
        return;
    }
    node._uvDecimatePresetBound = true;

    let syncing = false;
    const origPresetCallback = presetWidget.callback;
    const origTargetCallback = targetFacesWidget.callback;

    presetWidget.callback = function (value, ...args) {
        const res = origPresetCallback?.call(this, value, ...args);
        if (syncing) {
            return res;
        }
        const faces = PRESET_TO_FACES.get(String(value));
        if (faces !== undefined && Number(targetFacesWidget.value) !== faces) {
            syncing = true;
            try {
                targetFacesWidget.value = faces;
                origTargetCallback?.call(targetFacesWidget, faces);
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
            } finally {
                syncing = false;
            }
        }
        return res;
    };

    targetFacesWidget.callback = function (value, ...args) {
        const res = origTargetCallback?.call(this, value, ...args);
        if (syncing) {
            return res;
        }
        const num = Math.round(Number(value));
        const currentPresetFaces = PRESET_TO_FACES.get(String(presetWidget.value));
        if (currentPresetFaces !== undefined && num !== currentPresetFaces) {
            const matchedPreset = FACES_TO_PRESET.get(num) ?? "custom";
            syncing = true;
            try {
                presetWidget.value = matchedPreset;
                origPresetCallback?.call(presetWidget, matchedPreset);
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
            } finally {
                syncing = false;
            }
        }
        return res;
    };

    const initialFaces = PRESET_TO_FACES.get(String(presetWidget.value));
    if (initialFaces !== undefined && Number(targetFacesWidget.value) !== initialFaces) {
        targetFacesWidget.value = initialFaces;
    }
}

app.registerExtension({
    name: "comfyui.uv_decimate.preset_sync",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = onNodeCreated?.apply(this, arguments);
            bindPresetAndTargetFaces(this);
            return res;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const res = onConfigure?.apply(this, arguments);
            bindPresetAndTargetFaces(this);
            return res;
        };
    },
});
