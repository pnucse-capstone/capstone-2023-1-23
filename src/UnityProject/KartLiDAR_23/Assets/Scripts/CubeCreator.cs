using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class CubeCreator : MonoBehaviour
{
    public TMP_Text textButton;
    public Material realMaterial;

    List<GameObject> cubes = new List<GameObject>();

    // MeshFilter.mesh의 vertices와 triangles에 할당할 배열
    Vector3[] vertices = new Vector3[24];   // 6 faces * 4 vertices  = 24
    int[] triangles = new int[36];          // 6 faces * 3 vertices * 2 triangles = 36

    bool visibility = true;

    /* v8
     * 정육면체의 8개 정점의 순서는 위에서 바라보았을 때 기준으로 지정
     *  top | bottom
     *  0 1 | 4 5
     *  3 2 | 7 6
     */

    void SetMeshArray(ref Vector3[] v8)
    {
        int[][] vers =
        {
            new int[] { 0, 1, 2, 3 },   // top
            new int[] { 5, 4, 7, 6 },   // bottom
            new int[] { 3, 2, 6, 7 },   // front
            new int[] { 1, 0, 4, 5 },   // back
            new int[] { 0, 3, 7, 4 },   // left
            new int[] { 2, 1, 5, 6 }    // right
        };
        int[] tris = new int[6] // 시계 방향으로 순서 지정
        {
            0, 1, 2,    // upper right triangle
            3, 0, 2     // lower left
        };

        int vi = 0, ti = 0;
        for (int i = 0; i < 6; i++) // 6 faces
        {
            vertices[vi++] = v8[vers[i][0]];
            vertices[vi++] = v8[vers[i][1]];
            vertices[vi++] = v8[vers[i][2]];
            vertices[vi++] = v8[vers[i][3]];

            int[] v4 = new int[4]
            {
                vi - 4 + 0,
                vi - 4 + 1,
                vi - 4 + 2,
                vi - 4 + 3
            };

            triangles[ti++] = v4[tris[0]];
            triangles[ti++] = v4[tris[1]];
            triangles[ti++] = v4[tris[2]];
            triangles[ti++] = v4[tris[3]];
            triangles[ti++] = v4[tris[4]];
            triangles[ti++] = v4[tris[5]];
        }
    }

    public void CreateCube(ref Vector3[] v8, bool wall, bool real = false)
    {
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);

        SetMeshArray(ref v8);

        Mesh mesh = cube.GetComponent<MeshFilter>().mesh;
        mesh.Clear();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        string cubeName;

        if (real)
        {
            cube.GetComponent<Renderer>().material = realMaterial;
            cubeName = "real";

            if (wall)
                cubeName += "Wall";
            else
                cubeName += "Objt";
        }
        else
        {
            if (wall)
            {
                cube.GetComponent<Renderer>().material.color = Color.white;
                cubeName = "virtualWall";
            }
            else
            {
                cube.GetComponent<Renderer>().material.color = new Color(Random.value, Random.value, Random.value, 1f);
                cubeName = "virtualObjt";
            }
        }

        Destroy(cube.GetComponent<BoxCollider>());
        cube.gameObject.AddComponent<BoxCollider>();

        cube.name = cubeName;
        cubes.Add(cube);
    }

    public void ToggleVisibility()
    {
        if (visibility)
            textButton.text = "ON";
        else
            textButton.text = "OFF";

        visibility = !visibility;

        foreach (var cube in cubes)
            cube.GetComponent<Renderer>().enabled = !cube.GetComponent<Renderer>().enabled;
    }
}
