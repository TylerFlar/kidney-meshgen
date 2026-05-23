// Minimal Unity metadata loader scaffold.
// It does not force a GLB/OBJ importer. Assign imported mesh GameObjects to
// visualRoot/collisionRoot/stonesRoot, or extend this class with your importer.

using UnityEngine;

namespace KidneyMeshgen
{
    public class KidneyMeshgenSceneLoader : MonoBehaviour
    {
        [Header("JSON files")]
        public TextAsset manifestJson;
        public TextAsset runtimeSceneJson;

        [Header("Imported mesh roots")]
        public Transform visualRoot;
        public Transform collisionRoot;
        public Transform stonesRoot;

        [Header("Scope start")]
        public Transform scopeCameraOrTip;
        public bool convertSourceAxesToUnityYUp = true;

        public KidneyMeshgenManifest Manifest { get; private set; }
        public KidneyMeshgenSceneSpec SceneSpec { get; private set; }

        public void LoadMetadata()
        {
            if (manifestJson != null)
                Manifest = JsonUtility.FromJson<KidneyMeshgenManifest>(manifestJson.text);
            if (runtimeSceneJson != null)
                SceneSpec = JsonUtility.FromJson<KidneyMeshgenSceneSpec>(runtimeSceneJson.text);
        }

        public void ApplyScaleToImportedMeshes()
        {
            float scale = Manifest != null ? Manifest.unity_scale_to_meters : 0.001f;
            if (visualRoot != null) visualRoot.localScale = Vector3.one * scale;
            if (collisionRoot != null) collisionRoot.localScale = Vector3.one * scale;
            if (stonesRoot != null) stonesRoot.localScale = Vector3.one * scale;
        }

        public void PlaceScopeAtStart()
        {
            if (Manifest == null || Manifest.scope_start_pose == null || scopeCameraOrTip == null) return;
            float scale = Manifest.unity_scale_to_meters;
            Vector3 pos = KidneyMeshgenCoordinateUtils.MmToUnityMeters(Manifest.scope_start_pose.position_mm, scale, convertSourceAxesToUnityYUp);
            Vector3 look = KidneyMeshgenCoordinateUtils.MmToUnityMeters(Manifest.scope_start_pose.look_at_position_mm, scale, convertSourceAxesToUnityYUp);
            Vector3 up = KidneyMeshgenCoordinateUtils.VectorSourceToUnity(Manifest.scope_start_pose.up_vector, convertSourceAxesToUnityYUp);
            scopeCameraOrTip.position = pos;
            scopeCameraOrTip.rotation = KidneyMeshgenCoordinateUtils.LookAtRotation(pos, look, up);
        }

        private void Awake()
        {
            LoadMetadata();
        }
    }
}
