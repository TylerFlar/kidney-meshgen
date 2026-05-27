// Lightweight metadata classes only; mesh import is intentionally left to your chosen importer.

using System;
using UnityEngine;

namespace KidneyMeshgen
{
    [Serializable]
    public class KidneyMeshgenSceneSpec
    {
        public string schema;
        public string anatomy_id;
        public string units;
        public float unity_scale_to_meters = 0.001f;
        public SceneAssets assets;
        public StoneInfo[] stones;
        public ScopeModel scope_model;
        public SceneTask[] tasks;
    }

    [Serializable]
    public class SceneAssets
    {
        public string visual_lumen;
        public string collision_lumen;
        public string sdf_grid;
        public string stones;
        public string centerline_graph;
        public string waypoints;
        public string coverage_points;
        public string labels;
    }

    [Serializable]
    public class ScopeModel
    {
        public string model;
        public float outer_diameter_mm;
        public float tip_length_mm;
        public float fov_degrees;
        public float max_deflection_deg;
        public int camera_rate_hz;
        public int control_rate_hz;
    }

    [Serializable]
    public class SceneTask
    {
        public string id;
        public string goal;
        public string target_node;
        public string[] target_calyx_ids;
        public string[] stone_ids;
    }

    [Serializable]
    public class ScopeStartPose
    {
        public string node;
        public float[] position_mm;
        public string look_at_node;
        public float[] look_at_position_mm;
        public float[] forward_vector;
        public float[] up_vector;
        public float[] rotation_quaternion_xyzw;
    }

    [Serializable]
    public class CalyxTarget
    {
        public string id;
        public string region;
        public string cup_node;
        public float[] center_mm;
        public float approx_radius_mm;
    }

    [Serializable]
    public class StoneInfo
    {
        public string id;
        public string calyx_id;
        public string region;
        public float[] center_mm;
        public float radius_mm;
        public string mesh_file;
        public int label_id;
        public string material_class;
        public string state;
        public float[] color_rgba;
        public float roughness;
        public float specular;
        public float crystal_bump_strength;
        public float crystal_bump_distance_mm;
        public float crystal_bump_scale;
        public int fracture_plane_count;
        public int fragment_count;
        public float[] fragment_radius_mm;
        public float gravel_spread_mm;
    }

    [Serializable]
    public class KidneyMeshgenManifest
    {
        public string schema;
        public string anatomy_id;
        public int seed;
        public string units;
        public float unity_scale_to_meters = 0.001f;
        public string coordinate_system;
        public string pelvis_type;
        public CalyxTarget[] calyx_targets;
        public StoneInfo[] stones;
        public ScopeStartPose scope_start_pose;
    }

    public static class KidneyMeshgenCoordinateUtils
    {
        public static Vector3 MmToUnityMeters(float[] mm, float scale = 0.001f, bool unityYUp = true)
        {
            if (mm == null || mm.Length < 3) return Vector3.zero;
            if (unityYUp) return new Vector3(mm[0] * scale, mm[2] * scale, mm[1] * scale);
            return new Vector3(mm[0] * scale, mm[1] * scale, mm[2] * scale);
        }

        public static Vector3 VectorSourceToUnity(float[] v, bool unityYUp = true)
        {
            if (v == null || v.Length < 3) return Vector3.forward;
            if (unityYUp) return new Vector3(v[0], v[2], v[1]);
            return new Vector3(v[0], v[1], v[2]);
        }

        public static Quaternion LookAtRotation(Vector3 position, Vector3 lookAt, Vector3 up)
        {
            Vector3 forward = lookAt - position;
            if (forward.sqrMagnitude < 1e-8f) forward = Vector3.forward;
            if (up.sqrMagnitude < 1e-8f) up = Vector3.up;
            return Quaternion.LookRotation(forward.normalized, up.normalized);
        }
    }
}
