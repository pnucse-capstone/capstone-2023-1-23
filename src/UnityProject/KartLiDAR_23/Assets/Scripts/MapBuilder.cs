using OpenCvSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO.Enumeration;
using System.Linq;
using UnityEngine;

class PointProperty
{
    public float minH, maxH;

    public Vector2Int normalKey;
    public float weight;
}

class Point : PointProperty
{
    public Vector2 coord;
}

class PointManager
{
    List<PointProperty> segments = new List<PointProperty>();
    List<Vector2> coords = new List<Vector2>();

    public void AddPoint(Vector3 a, Vector3 b, float minH, float maxH, Vector2Int nKey, float weight)
    {
        coords.Add(new Vector2(a.x, a.z));
        coords.Add(new Vector2(b.x, b.z));

        segments.Add(new PointProperty
        {
            minH = minH,
            maxH = maxH,
            normalKey = nKey,
            weight = weight
        });
    }

    public int GetCoordNum() { return coords.Count; }
    public Vector2 GetCoordinate(int i) { return coords[i]; }
    public float GetMinHeight(int i) { return segments[i / 2].minH; }
    public float GetMaxHeight(int i) { return segments[i / 2].maxH; }
    public Vector2Int GetNormalKey(int i) { return segments[i / 2].normalKey; }
    public float GetWeight(int i) { return segments[i / 2].weight; }

    public void ChangeCoordinate(int i, Vector2 newPoint)
    {
        coords[i] = newPoint;
    }

    public int CompareYX(int iA, int iB)
    {
        Vector2 pA = GetCoordinate(iA);
        Vector2 pB = GetCoordinate(iB);

        if (pA.y == pB.y)
            return pA.x < pB.x ? -1 : 1;
        return pA.y < pB.y ? -1 : 1;
    }
    public int CompareClockwise(int o, int iA, int iB)
    {
        Vector2 origin = GetCoordinate(o);
        Vector2 pA = GetCoordinate(iA);
        Vector2 pB = GetCoordinate(iB);

        return Mathf.Atan2(pA.y - origin.y, pA.x - origin.x).CompareTo(Mathf.Atan2(pB.y - origin.y, pB.x - origin.x));
    }
    public bool IsCCW(int iA, int iB, int iC)
    {
        Vector2 pA = GetCoordinate(iA);
        Vector2 pB = GetCoordinate(iB);
        Vector2 pC = GetCoordinate(iC);

        return (pA.x * pB.y + pB.x * pC.y + pC.x * pA.y) - (pA.y * pB.x + pB.y * pC.x + pC.y * pA.x) > 0;
    }

    public void Translate(Vector2 o)
    {
        for (int i = 0; i < coords.Count; i++)
            coords[i] -= o;
    }
    public void Rotate(float radian)
    {
        for (int i = 0; i < coords.Count; i++)
        {
            Vector2 coord = coords[i];
            float x = coord.x, y = coord.y;

            coord.x = x * Mathf.Cos(radian) - y * Mathf.Sin(radian);
            coord.y = x * Mathf.Sin(radian) + y * Mathf.Cos(radian);
            coords[i] = coord;
        }
    }
}

public class MapBuilder : MonoBehaviour
{
    public string fileName = "filePath/filename.txt";
    public float wallThickness = 0.1f;
    public float rangeToExclude = 1;
    public float scaleConstant = 10f;

    public CubeCreator wallCreator;
    public CubeCreator objtCreator;

    public GameObject cameraTarget;

    const float tolerance = 0.0001f;    // float 계산 오차 허용 범위

    PointManager pointManager = new PointManager();

    List<int> pointsToExclude = new List<int>();
    Dictionary<Vector2Int, float> normalMaxWeight = new Dictionary<Vector2Int, float>();

    Vector2[] wallVertices = new Vector2[4];

    /********************************************************************************************/

    Vector3 CalculateNormal(ref Vector3[] p)
    {
        Vector3 ab = p[1] - p[0];
        Vector3 ac = p[2] - p[0];
        return Vector3.Cross(ab, ac).normalized;
    }

    void ReadData()
    {
        string[] contents = System.IO.File.ReadAllLines(fileName);

        // txt file 1 line == 1 face
        for (int l = 0; l < contents.Length; l++)
        {
            string[] txt = contents[l].Split(',');  // 1 face == 4 points == 12 numbers
            Vector3[] rawP = new Vector3[4];

            for (int xyzi = 0; xyzi < 12; xyzi++)
            {
                if (xyzi % 3 == 2)
                {
                    float x = float.Parse(txt[xyzi - 2]) * scaleConstant;
                    float z = float.Parse(txt[xyzi - 1]) * scaleConstant;
                    float y = float.Parse(txt[xyzi]) * scaleConstant;   // height

                    if (y <= 1f && y >= -1f) y = 0f;

                    rawP[xyzi / 3] = new Vector3(x, y, z);
                }
            }

            /* 면이 회전한 경우 보정 필요
             *  면(벽)을 이루는 4개 점 rawP은 이웃하게 나열되어 있으므로 높이가 비슷한 두 점씩 묶으면 01,23 또는 12,30의 2가지 경우
             *    01,23          12,30
             *  0 ------ 1     1 ------ 2
             *  |        | 또는 |        |
             *  3 ------ 2     0 ------ 3
             */
            int firstIdx = 0;   // 01,23
            if (Mathf.Abs(rawP[0].y - rawP[1].y) > Mathf.Abs(rawP[1].y - rawP[2].y))
                firstIdx++;     // 12,30

            // 면의 회전축이 x축 또는 z축과 평행하거나 수직인 경우 rawP 수정
            Vector3 normal = CalculateNormal(ref rawP);
            if (normal.y != 0 || rawP[firstIdx].y != rawP[firstIdx + 1].y)
            {
                // 장애물의 top(위쪽 면)인 경우 데이터 저장 X
                if (normal.y > 0.7f || normal.y < -0.7f)
                    continue;

                for (int i = 0; i < 2; i++)
                {
                    /* 회전한 면을 xz 평면으로 사영시켰을 때 생기는 점을 4개 -> 2개로 줄이기 위해
                     * 가까운 2개 점A, B을 평균내어 새로운 1개 점 생성
                     *  i = 0: 03 또는 10
                     *  i = 1: 12 또는 23
                     */
                    int idxA = firstIdx + i, idxB = idxA + 1;
                    if (i == 0 && idxA == 0) idxB = 3;
                    else if (i == 0 && idxA == 1) idxB = 0;

                    float maxH = Mathf.Max(rawP[idxA].y, rawP[idxB].y);
                    //float maxH = (rawP[idxA].y + rawP[idxB].y) / 2f;
                    //float maxH = (Mathf.Max(rawP[idxA].y, rawP[idxB].y) + (rawP[idxA].y + rawP[idxB].y) / 2f) / 2f;
                    float minH = Mathf.Min(rawP[idxA].y, rawP[idxB].y, 0f);
                    //float minH = Mathf.Min(rawP[idxA].y, rawP[idxB].y);
                    if (minH < 0)
                        minH = 0;

                    rawP[idxA] = new Vector3(
                        (rawP[idxA].x + rawP[idxB].x) / 2f,
                        maxH,
                        (rawP[idxA].z + rawP[idxB].z) / 2f);
                    rawP[idxB] = new Vector3(
                        (rawP[idxA].x + rawP[idxB].x) / 2f,
                        minH,
                        (rawP[idxA].z + rawP[idxB].z) / 2f);
                }
            }
            // ========== rawP 수정 완료

            float weight = (rawP[firstIdx] - rawP[firstIdx + 2]).sqrMagnitude;
            Vector2Int normalKey = new Vector2Int(Mathf.RoundToInt(Mathf.Abs(normal.x)), Mathf.RoundToInt(Mathf.Abs(normal.z)));

            if (!normalMaxWeight.ContainsKey(normalKey))
                normalMaxWeight.Add(normalKey, weight);
            else if (normalMaxWeight[normalKey] < weight)
                normalMaxWeight[normalKey] = weight;

            pointManager.AddPoint(rawP[0], rawP[2], Mathf.Min(rawP[0].y, rawP[2].y), Mathf.Max(rawP[0].y, rawP[2].y), normalKey, weight);
        }
    }

    void SetPointsToExclude()
    {
        List<int> idx = Enumerable.Range(0, pointManager.GetCoordNum()).ToList();
        HashSet<Vector2> uniqueHash = new HashSet<Vector2>();
        foreach (int i in idx)
        {
            if (!uniqueHash.Add(pointManager.GetCoordinate(i)))
                pointsToExclude.Add(i);
        }
    }

    void SetPointsToExclude(ref List<int> idx)
    {
        pointsToExclude.Clear();
        HashSet<Vector2> uniqueHash = new HashSet<Vector2>();

        foreach (int i in idx)
        {
            if (!uniqueHash.Add(pointManager.GetCoordinate(i)))
                pointsToExclude.Add(i);
        }
    }

    void SetPointsToExcludeForObject()
    {
        pointsToExclude.Clear();

        int segmentNum = pointManager.GetCoordNum() / 2;
        for (int i = 0; i < segmentNum; i++)
        {
            Vector2 pA = pointManager.GetCoordinate(i * 2);
            Vector2 pB = pointManager.GetCoordinate(i * 2 + 1);
            float halfAB = (pA - pB).magnitude / 2f;

            // 벽 4방향에 대해 내부에 위치한 점이면 true
            bool northA = pA.y < wallVertices[0].y - rangeToExclude;
            bool eastA = pA.x < wallVertices[1].x - rangeToExclude;
            bool southA = pA.y > wallVertices[2].y + rangeToExclude;
            bool westA = pA.x > wallVertices[0].x + rangeToExclude;

            bool northB = pB.y < wallVertices[0].y - rangeToExclude;
            bool eastB = pB.x < wallVertices[1].x - rangeToExclude;
            bool southB = pB.y > wallVertices[2].y + rangeToExclude;
            bool westB = pB.x > wallVertices[0].x + rangeToExclude;

            bool excludeA = !(northA && eastA && southA && westA);
            bool excludeB = !(northB && eastB && southB && westB);

            float northDist = 0f, eastDist = 0f, southDist = 0f, westDist = 0f;

            if (excludeA && !excludeB)  // A 점만 제외
            {
                if (!northA) northDist = Mathf.Abs(wallVertices[0].y - pA.y);
                if (!eastA) eastDist = Mathf.Abs(wallVertices[1].x - pA.x);
                if (!southA) southDist = Mathf.Abs(wallVertices[2].y - pA.y);
                if (!westA) westDist = Mathf.Abs(wallVertices[0].x - pA.x);

                float rangeToExclude_ = Mathf.Max(northDist, eastDist, southDist, westDist, rangeToExclude);

                if ((!northA && wallVertices[0].y - rangeToExclude_ - pB.y < halfAB) ||
                    (!eastA && wallVertices[1].x - rangeToExclude_ - pB.x < halfAB) ||
                    (!southA && pB.y - wallVertices[2].y - rangeToExclude_ < halfAB) ||
                    (!westA && pB.x - wallVertices[0].x - rangeToExclude_ < halfAB))
                    excludeB = true;
            }
            else if (!excludeA && excludeB)  // B 점만 제외
            {
                if (!northB) northDist = Mathf.Abs(wallVertices[0].y - pB.y);
                if (!eastB) eastDist = Mathf.Abs(wallVertices[1].x - pB.x);
                if (!southB) southDist = Mathf.Abs(wallVertices[2].y - pB.y);
                if (!westB) westDist = Mathf.Abs(wallVertices[0].x - pB.x);

                float rangeToExclude_ = Mathf.Max(northDist, eastDist, southDist, westDist, rangeToExclude);

                if ((!northB && wallVertices[0].y - rangeToExclude_ - pA.y < halfAB) ||
                    (!eastB && wallVertices[1].x - rangeToExclude_ - pA.x < halfAB) ||
                    (!southB && pA.y - wallVertices[2].y - rangeToExclude_ < halfAB) ||
                    (!westB && pA.x - wallVertices[0].x - rangeToExclude_ < halfAB))
                    excludeA = true;
            }

            if (excludeA && excludeB)
            {
                pointsToExclude.Add(i * 2);
                pointsToExclude.Add(i * 2 + 1);
                /*
                GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.transform.position = new Vector3(pointManager.GetCoordinate(i * 2).x, 0, pointManager.GetCoordinate(i * 2).y);
                sphere.transform.localScale = Vector3.one * 0.3f;

                GameObject s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                s.transform.position = new Vector3(pointManager.GetCoordinate(i * 2 + 1).x, 0, pointManager.GetCoordinate(i * 2 + 1).y);
                s.transform.localScale = Vector3.one * 0.3f;
                */
            }
        }
    }

    List<int> ConvexHull()
    {
        List<int> idx = Enumerable.Range(0, pointManager.GetCoordNum()).ToList();
        SetPointsToExclude();
        return ConvexHull(ref idx);
    }

    List<int> ConvexHull(ref List<int> idx)
    {
        idx.RemoveAll(i => pointsToExclude.Contains(i));

        idx.Sort(pointManager.CompareYX);
        int originIdx = idx[0];
        idx.Sort(delegate (int a, int b) { return pointManager.CompareClockwise(originIdx, a, b); });

        Stack<int> hull = new Stack<int>();
        int first = 0, second = 1, next = 2;
        hull.Push(first);
        hull.Push(second);

        while (next < idx.Count)
        {
            while (hull.Count >= 2)
            {
                second = hull.Pop();
                first = hull.Peek();

                if (pointManager.IsCCW(idx[first], idx[second], idx[next]))
                {
                    hull.Push(second);
                    break;
                }
            }

            hull.Push(next++);
        }

        List<int> hullIdx = new List<int>();
        while (hull.Count > 0)
            hullIdx.Add(idx[hull.Pop()]);
        return hullIdx;
    }

    List<Point> MergeHullTo4Points(bool forWall, ref List<int> hull)
    {
        List<Point> weighted4 = new List<Point>(hull.Count);
        foreach (int i in hull)
        {
            Point p = new Point();
            p.coord = pointManager.GetCoordinate(i);

            int idx = i;
            if (!forWall && i / 2 >= pointManager.GetCoordNum())
            {
                idx = 0;
                while (hull[idx] / 2 >= pointManager.GetCoordNum()) { idx++; }
                idx = hull[idx];
            }

            p.minH = pointManager.GetMinHeight(idx);
            p.maxH = pointManager.GetMaxHeight(idx);
            p.normalKey = pointManager.GetNormalKey(idx);
            p.weight = pointManager.GetWeight(idx);

            weighted4.Add(p);
        }

        while (weighted4.Count > 4)
        {
            float minDistSqr = float.MaxValue;
            int[] idxToMerge = new int[2];
            // 병합할 때 weighted4의 idx가 더 작은 곳으로 병합하고 큰 곳은 삭제

            for (int i = 0; i < weighted4.Count; i++)
            {
                int small = i, large = i + 1;
                if (large == weighted4.Count)
                {
                    large = small;
                    small = 0;
                }

                float distSqr = (weighted4[small].coord - weighted4[large].coord).sqrMagnitude;
                if (distSqr < minDistSqr)
                {
                    minDistSqr = distSqr;
                    idxToMerge[0] = small;
                    idxToMerge[1] = large;
                }
            }

            float weightDiffSmall = normalMaxWeight[weighted4[idxToMerge[0]].normalKey] - weighted4[idxToMerge[0]].weight;
            float weightDiffLarge = normalMaxWeight[weighted4[idxToMerge[1]].normalKey] - weighted4[idxToMerge[1]].weight;

            // weightDiff가 같으면 두 점을 평균 내어 새로운 점 생성
            if (weightDiffSmall == weightDiffLarge)
            {
                weighted4[idxToMerge[0]].coord = (weighted4[idxToMerge[0]].coord + weighted4[idxToMerge[1]].coord) / 2f;
                weighted4[idxToMerge[0]].minH = (weighted4[idxToMerge[0]].minH + weighted4[idxToMerge[1]].minH) / 2f;
                weighted4[idxToMerge[0]].maxH = (weighted4[idxToMerge[0]].maxH + weighted4[idxToMerge[1]].maxH) / 2f;
            }
            else if ((forWall && weightDiffSmall > weightDiffLarge) || (!forWall && weightDiffSmall < weightDiffLarge))
                weighted4[idxToMerge[0]] = weighted4[idxToMerge[1]];

            weighted4.RemoveAt(idxToMerge[1]);
        }

        return weighted4;
    }

    void Translate(bool forWall, Vector2 o, ref List<Point> points)
    {
        for (int i = 0; i < points.Count; i++)
            points[i].coord -= o;

        if (forWall)
            pointManager.Translate(o);
    }

    void Rotate(bool forWall, float radian, ref List<Point> points)
    {
        radian *= -1f;
        //Debug.Log("[Rotate] 변환하는 각도: " + radian * 180.0 / Mathf.PI);

        for (int i = 0; i < points.Count; i++)
        {
            Vector2 coord = points[i].coord;
            float x = coord.x, y = coord.y;

            coord.x = x * Mathf.Cos(radian) - y * Mathf.Sin(radian);
            coord.y = x * Mathf.Sin(radian) + y * Mathf.Cos(radian);
            points[i].coord = coord;
        }

        if (forWall)
            pointManager.Rotate(radian);
    }

    (bool have, Vector2 intersection) Intersection(Vector2 a1, Vector2 b1, Vector2 a2, Vector2 b2)
    {
        Vector2 intersection = Vector2.zero;

        float denominator = (a1.x - b1.x) * (a2.y - b2.y) - (a1.y - b1.y) * (a2.x - b2.x);
        if (denominator > -tolerance && denominator < tolerance)
            return (false, intersection);

        float constant = ((b2.x - a2.x) * (a1.y - a2.y) - (b2.y - a2.y) * (a1.x - a2.x)) / denominator;
        return (true, new Vector2(a1.x + constant * (b1.x - a1.x), a1.y + constant * (b1.y - a1.y)));
    }

    Vector2 Intersection(Vector2 p, Vector2 a, Vector2 b)
    {
        // y = mx + b
        float mSeg = (b.y - a.y) / (a.x - b.x), bSeg = a.y - mSeg * a.x;
        float mNormal = -1 / mSeg, bNormal = p.y - mSeg * p.x;

        return new Vector2((bNormal - bSeg) / (mSeg - mNormal),
                           mSeg * (bNormal - bSeg) / (mSeg - mNormal) + bSeg);
    }

    /// <summary>
    /// 선분 1과 선분 2의 교차점이 선분 내에 존재하면 1 반환, 존재하지 않으면 -1 반환,
    /// 평행하는 경우 0 반환
    /// </summary>
    /// <param name="a1">선분 1의 점 A</param>
    /// <param name="b1">선분 1의 점 B</param>
    /// <param name="a2">선분 2의 점 A</param>
    /// <param name="b2">선분 2의 점 B</param>
    /// <returns></returns>
    int CheckIntersection(Vector2 a1, Vector2 b1, Vector2 a2, Vector2 b2)
    {
        (bool hasIntersection, Vector2 intersection) = Intersection(a1, b1, a2, b2);
        if (!hasIntersection) return 0;

        float dist1 = (a1 - b1).magnitude;
        float dist1AI = (a1 - intersection).magnitude;
        float dist1IB = (b1 - intersection).magnitude;

        float dist2 = (a2 - b2).magnitude;
        float dist2AI = (a2 - intersection).magnitude;
        float dist2IB = (b2 - intersection).magnitude;

        if (Mathf.Abs(dist1 - (dist1AI + dist1IB)) < tolerance && Mathf.Abs(dist2 - (dist2AI + dist2IB)) < tolerance)
            return 1;
        else
            return -1;
    }

    float ShortestDistPointToSegment(int pi, int si)
    {
        Vector2 p = pointManager.GetCoordinate(pi);
        Vector2 a = pointManager.GetCoordinate(si), b = pointManager.GetCoordinate(si + 1);

        Vector2 intersection = Intersection(p, a, b);

        if (CheckIntersection(p, intersection, a, b) == 1)
            return (p - intersection).magnitude;
        else
            return float.MaxValue;
    }

    float ShortestDistBetweenSegments(int s1i, int s2i)
    {
        float dist = 0f;

        Vector2 a1 = pointManager.GetCoordinate(s1i), b1 = pointManager.GetCoordinate(s1i + 1);
        Vector2 a2 = pointManager.GetCoordinate(s2i), b2 = pointManager.GetCoordinate(s2i + 1);

        int infoIntersection = CheckIntersection(a1, b1, a2, b2);
        if (infoIntersection != 1)  // 선분 내에 교차점이 존재하지 않음
        {
            float a1s2 = ShortestDistPointToSegment(s1i, s2i);
            float b1s2 = ShortestDistPointToSegment(s1i + 1, s2i);
            float a2s1 = ShortestDistPointToSegment(s2i, s1i);
            float b2s1 = ShortestDistPointToSegment(s2i + 1, s1i);

            float a1a2 = (a1 - a2).magnitude;
            float a1b2 = (a1 - b2).magnitude;
            float b1a2 = (b1 - a2).magnitude;
            float b1b2 = (b1 - b2).magnitude;

            dist = Mathf.Min(a1s2, b1s2, a2s1, b2s1, a1a2, a1b2, b1a2, b1b2);
        }

        return dist;
    }

    List<List<int>> SetObjectGroups()
    {
        List<int> idx = Enumerable.Range(0, pointManager.GetCoordNum()).ToList();
        idx.RemoveAll(i => pointsToExclude.Contains(i));

        // dist: 선분 사이 최단 거리, segmentIdx1: 선분 1 점 A idx, segmentIdx2: 선분 2 점 A idx
        List<(float dist, int segmentIdx1, int segmentIdx2)> distList = new List<(float dist, int segmentIdx1, int segmentIdx2)>();
        for (int i = 0; i < idx.Count; i += 2)
        {
            for (int j = i + 2; j < idx.Count; j += 2)
            {
                float d = ShortestDistBetweenSegments(idx[i], idx[j]);
                distList.Add((d, idx[i], idx[j]));
            }
        }
        distList.Sort();    // dist 오름차순 정렬

        // key: 점 idx, value: key 점이 속한 장애물 그룹의 리더 점 idx
        Dictionary<int, int> pointGroup = new Dictionary<int, int>();
        for (int i = 0; i < distList.Count; i++)
        {
            int idx1 = distList[i].segmentIdx1, idx2 = distList[i].segmentIdx2;
            bool contain1 = pointGroup.ContainsKey(idx1);
            bool contain2 = pointGroup.ContainsKey(idx2);

            if (contain1 && contain2)
                continue;
            else if (contain1)
            {
                pointGroup.Add(idx2, pointGroup[idx1]);
                pointGroup.Add(idx2 + 1, pointGroup[idx1]);
            }
            else if (contain2)
            {
                pointGroup.Add(idx1, pointGroup[idx2]);
                pointGroup.Add(idx1 + 1, pointGroup[idx2]);
            }
            else
            {
                int minIdx = Mathf.Min(idx1, idx2);
                pointGroup.Add(idx1, minIdx);
                pointGroup.Add(idx1 + 1, minIdx);
                pointGroup.Add(idx2, minIdx);
                pointGroup.Add(idx2 + 1, minIdx);
            }
        }

        // key: 장애물 그룹의 리더 점 idx, value: 그룹에 포함된 점 idx List
        Dictionary<int, List<int>> objectDict = new Dictionary<int, List<int>>();
        foreach (var pair in pointGroup)
        {
            if (!objectDict.ContainsKey(pair.Value))
                objectDict.Add(pair.Value, new List<int>());
            objectDict[pair.Value].Add(pair.Key);
        }

        // 장애물 그룹에 포함된 점 idx들의 List를 List에 저장
        List<List<int>> objectGroups = new List<List<int>>();
        foreach (var pair in objectDict)
        {
            pair.Value.Sort();

            int groupNum = pair.Value.Count;
            if (groupNum <= 6)
            {
                float[] x = new float[groupNum];
                float[] y = new float[groupNum];

                for (int i = 0; i < groupNum; i++)
                {
                    Vector2 p = pointManager.GetCoordinate(pair.Value[i]);
                    x[i] = p.x;
                    y[i] = p.y;
                }

                Array.Sort(x);
                Array.Sort(y);

                // 장애물 그룹 점에 기반하여 새로 만들어지는 4개 점 중 기존 점과의 shortest dist가 가장 긴 1개의 점만 장애물 그룹에 추가
                float minX = x[0], maxX = x[groupNum - 1];
                float minY = y[0], maxY = y[groupNum - 1];

                (float minD, Vector2 p)[] newP = new (float minD, Vector2 p)[4];
                for (int i = 0; i < 4; i++)
                {
                    newP[i].minD = float.MaxValue;
                    if (i == 0) newP[i].p = new Vector2(minX, maxY);
                    else if (i == 1) newP[i].p = new Vector2(maxX, maxY);
                    else if (i == 2) newP[i].p = new Vector2(maxX, minY);
                    else newP[i].p = new Vector2(minX, minY);

                    foreach (int iObjt in pair.Value)
                    {
                        float dist = (pointManager.GetCoordinate(iObjt) - newP[0].p).magnitude;
                        if (newP[i].minD > dist) newP[i].minD = dist;
                    }
                }
                int idxToAdd = 0;
                for (int i = 0; i < 4; i++)
                {
                    if (newP[idxToAdd].minD > newP[i].minD) idxToAdd = i;
                }

                Vector3 pointToAdd = new Vector3(newP[idxToAdd].p.x, 0, newP[idxToAdd].p.y);
                pointManager.AddPoint(pointToAdd, pointToAdd,
                    pointManager.GetMinHeight(pair.Value[0]), pointManager.GetMaxHeight(pair.Value[0]),
                    pointManager.GetNormalKey(pair.Value[0]), pointManager.GetWeight(pair.Value[0]));
                pair.Value.Add(pointManager.GetCoordNum() - 1);
            }

            objectGroups.Add(pair.Value);
        }

        Debug.Log("[MapBuilder: SetObjectGroups] 장애물 개수: " + objectGroups.Count);
        return objectGroups;
    }

    /// <summary>
    /// Point 4개를 이용하여 만들 수 있는 가장 큰 사각형과 가장 작은 사각형의 평균 사각형을 구함
    /// Point idx 배치
    /// 0 1
    /// 3 2
    /// </summary>
    /// <param name="first">ResizeAvgRectangle 첫 실행 여부</param>
    void ResizeAvgRectangle(bool first, ref List<Point> points)
    {
        float minX = (points[0].coord.x + points[3].coord.x) / 2f;
        Translate(false, new Vector2(minX, 0), ref points);

        float maxX = 0f, maxZ = 0f;
        if (first)
        {
            maxX = (points[1].coord.x + points[2].coord.x) / 2f;
            maxZ = (points[0].coord.y + points[1].coord.y) / 2f;
        }
        else
        {
            maxX = Mathf.Min(points[1].coord.x, points[2].coord.x);
            maxZ = Mathf.Min(points[0].coord.y, points[1].coord.y);
        }

        // 0 1
        // 3 2
        points[0].coord = new Vector2(0, maxZ);
        points[1].coord = new Vector2(maxX, maxZ);
        points[2].coord = new Vector2(maxX, 0);
        points[3].coord = new Vector2(0, 0);


    }

    void CreateCube(ref Vector3[] v8, bool wall, bool real = false)
    {
        if (wall)
            wallCreator.CreateCube(ref v8, wall, real);
        else
            objtCreator.CreateCube(ref v8, wall, real);
    }

    void CreateWalls(ref List<Point> points)
    {
        Point2f[] srcP = new Point2f[4]
        {
            new Point2f(points[0].coord.x, points[0].coord.y),
            new Point2f(points[1].coord.x, points[1].coord.y),
            new Point2f(points[2].coord.x, points[2].coord.y),
            new Point2f(points[3].coord.x, points[3].coord.y)
        };

        ResizeAvgRectangle(true, ref points);
        for (int i = 0; i < 4; i++)
            wallVertices[i] = points[i].coord;
        Debug.Log("[MapBuilder: CreateWalls] 바닥: " + wallVertices[1]);

        Point2f[] dstP = new Point2f[4]
        {
            new Point2f(wallVertices[0].x, wallVertices[0].y),
            new Point2f(wallVertices[1].x, wallVertices[1].y),
            new Point2f(wallVertices[2].x, wallVertices[2].y),
            new Point2f(wallVertices[3].x, wallVertices[3].y)
        };

        Mat perspectiveMatrix = Cv2.GetPerspectiveTransform(srcP, dstP);

        int numP = pointManager.GetCoordNum();
        Point2f[] allP = new Point2f[numP];
        for (int i = 0; i < numP; i++)
        {
            Vector2 p = pointManager.GetCoordinate(i);
            allP[i] = new Point2f(p.x, p.y);
        }

        Point2f[] transformedPoints = Cv2.PerspectiveTransform(allP, perspectiveMatrix);

        for (int i = 0; i < numP; i++)
        {
            Vector2 p = new Vector2(transformedPoints[i].X, transformedPoints[i].Y);
            pointManager.ChangeCoordinate(i, p);
        }

        Vector3[] v8 = new Vector3[8];
        Vector3 thickness, maxH3, minH3;

        ///*
        float maxHS = 0, minHS = 0;
        for (int i = 0; i < 4; i++)
        {
            maxHS += points[i].maxH;
            minHS += points[i].minH;
        }
        maxH3 = new Vector3(0, maxHS / 4f, 0);
        minH3 = new Vector3(0, minHS / 4f, 0);

        Debug.Log("[MapBuilder: CreateWalls] 벽 높이: " + minH3.y + " ~ " + maxH3.y);

        // ====== 4개 벽면 각각의 큐브 생성
        for (int i = 0; i < 4; i++)
        {
            int cur = i, nxt = cur + 1;
            if (cur == 3)
                nxt = 0;

            Vector3 curP = new Vector3(points[cur].coord.x, 0, points[cur].coord.y);
            Vector3 nxtP = new Vector3(points[nxt].coord.x, 0, points[nxt].coord.y);

            thickness = Vector3.zero;
            if (cur == 0)
                thickness.z = wallThickness;
            else if (cur == 1)
                thickness.x = wallThickness;
            else if (cur == 2)
                thickness.z = -wallThickness;
            else
                thickness.x = -wallThickness;

            // bottom
            v8[4] = curP + minH3 + thickness;
            v8[5] = nxtP + minH3 + thickness;
            v8[6] = nxtP + minH3;
            v8[7] = curP + minH3;

            // top
            v8[0] = v8[4] + maxH3;
            v8[1] = v8[5] + maxH3;
            v8[2] = v8[6] + maxH3;
            v8[3] = v8[7] + maxH3;

            CreateCube(ref v8, true);
        }
        // ====== 벽면 완료

        // ====== 바닥면 큐브 생성
        thickness = new Vector3(0, wallThickness, 0);

        // top
        v8[0] = new Vector3(wallVertices[0].x, 0, wallVertices[0].y);
        v8[1] = new Vector3(wallVertices[1].x, 0, wallVertices[1].y);
        v8[2] = new Vector3(wallVertices[2].x, 0, wallVertices[2].y);
        v8[3] = new Vector3(wallVertices[3].x, 0, wallVertices[3].y);

        // bottom
        v8[4] = v8[0] - thickness;
        v8[5] = v8[1] - thickness;
        v8[6] = v8[2] - thickness;
        v8[7] = v8[3] - thickness;

        CreateCube(ref v8, true);
        // ====== 바닥면 완료
    }

    void CreateObjt(ref List<Point> points, Vector2 rotO, Vector2 transO)
    {
        ResizeAvgRectangle(true, ref points);
        Vector2 size = points[1].coord;
        Debug.Log("[MapBuilder: CreateObjt] 장애물 크기: " + size);

        Rotate(false, -Mathf.Atan2(rotO.y, rotO.x), ref points);
        Translate(false, new Vector2(-transO.x, -transO.y), ref points);

        // 장애물이 벽을 넘어가는 경우 추가 보정
        bool oneMore = false;
        for (int i = 0; i < 4; i++)
        {
            Vector2 p = points[i].coord;
            p.x = Mathf.Clamp(p.x, 0, wallVertices[1].x);
            p.y = Mathf.Clamp(p.y, 0, wallVertices[0].y);

            if (p.x != points[i].coord.x || p.y != points[i].coord.y)
                oneMore = true;
        }

        if (oneMore)
        {
            Translate(false, transO, ref points);
            Rotate(false, Mathf.Atan2(rotO.y, rotO.x), ref points);
            ResizeAvgRectangle(false, ref points);
            Rotate(false, -Mathf.Atan2(rotO.y, rotO.x), ref points);
            Translate(false, new Vector2(-transO.x, -transO.y), ref points);

            for (int i = 0; i < 4; i++)
            {
                if (points[i].coord.x < 0)
                    Translate(false, new Vector2(points[i].coord.x, 0), ref points);
                if (points[i].coord.x > wallVertices[1].x)
                    Translate(false, new Vector2(points[i].coord.x - wallVertices[1].x, 0), ref points);

                if (points[i].coord.y < 0)
                    Translate(false, new Vector2(0, points[i].coord.y), ref points);
                if (points[i].coord.y > wallVertices[0].y)
                    Translate(false, new Vector2(0, points[i].coord.y - wallVertices[0].y), ref points);
            }
        }
        // ====== 추가 보정 완료

        Vector3[] v8 = new Vector3[8];
        Vector3 maxH3, minH3;
        Vector2 center = Vector2.zero;

        float maxHS = 0, minHS = 0;
        for (int i = 0; i < 4; i++)
        {
            center += points[i].coord;
            maxHS += points[i].maxH;
            minHS += points[i].minH;
        }
        maxH3 = new Vector3(0, maxHS / 4f, 0);
        minH3 = new Vector3(0, minHS / 4f, 0);
        center /= 4;
        Debug.Log("[MapBuilder: CreateObjt] 벽 높이: " + minH3.y + " ~ " + maxH3.y);

        Vector3[] points3 = new Vector3[4];
        for (int i = 0; i < 4; i++)
            points3[i] = new Vector3(points[i].coord.x, 0, points[i].coord.y);
        //Debug.Log("[MapBuilder: CreateObjt] 장애물 바닥: " + points3[1]);
        Debug.Log("[MapBuilder: CreateObjt] 장애물 중심: " + center);
        Debug.Log("[MapBuilder: CreateObjt] 장애물 x: " + (center.x - size.x / 2f) + ", " + (wallVertices[1].x - center.x - size.x / 2f));
        Debug.Log("[MapBuilder: CreateObjt] 장애물 y: " + (center.y - size.y / 2f) + ", " + (wallVertices[1].y - center.y - size.y / 2f));

        // bottom
        v8[4] = points3[0] + minH3;
        v8[5] = points3[1] + minH3;
        v8[6] = points3[2] + minH3;
        v8[7] = points3[3] + minH3;

        // top
        v8[0] = points3[0] + maxH3;
        v8[1] = points3[1] + maxH3;
        v8[2] = points3[2] + maxH3;
        v8[3] = points3[3] + maxH3;

        CreateCube(ref v8, false);
    }

    /// <summary>
    /// 장애물의 4개 선분 중 회전의 기준이 될 선분이 idx 2, 3으로 가도록 정렬
    /// </summary>
    void SortObjtPoints(ref List<Point> objt4)
    {
        float minDiff = float.MaxValue;
        int bestline = 0;

        for (int i = 0; i < 4; i++)
        {
            int cur, cur_, pre, pre_, nxt, nxt_;
            pre = (i + 3) % 4;
            pre_ = i; cur = pre_;
            cur_ = (i + 1) % 4; nxt = cur_;
            nxt_ = (i + 2) % 4;

            Vector2 curGrd = objt4[cur_].coord - objt4[cur].coord;
            Vector2 befGrd = objt4[pre_].coord - objt4[pre].coord;
            Vector2 aftGrd = objt4[nxt_].coord - objt4[nxt].coord;

            float angleCur = curGrd.y / curGrd.x;
            float angleBef = befGrd.y / befGrd.x;
            float angleAft = aftGrd.y / aftGrd.x;

            if (curGrd.x == 0) angleCur = -1;
            if (befGrd.x == 0) angleBef = -1;
            if (aftGrd.x == 0) angleAft = -1;

            float Diff = Mathf.Abs(-1 - angleCur * angleBef) + Mathf.Abs(-1 - angleCur * angleAft);
            if (Diff < minDiff)
            {
                minDiff = Diff;
                bestline = cur;
            }
        }

        Vector2[] pointSort = new Vector2[4];
        int add = 2;
        if (bestline == 1)
            add = -1;
        else if (bestline == 2)
            add = 0;
        else if (bestline == 3)
            add = 1;

        for (int i = 0; i < 4; i++)
        {
            int j = i + add;
            if (j >= 4 || j < 0)
                j = (j + 4) % 4;
            pointSort[i] = objt4[j].coord;
        }

        for (int i = 0; i < 4; i++)
            objt4[i].coord = pointSort[i];
    }

    void Start()
    {
        fileName = PlayerPrefs.GetString("fileName");
        wallThickness = PlayerPrefs.GetFloat("wallThickness");
        scaleConstant = (float)PlayerPrefs.GetInt("scaleConstant");

        ReadData();

        List<int> wallHull = ConvexHull();
        List<Point> wall4 = MergeHullTo4Points(true, ref wallHull);
        Translate(true, wall4[3].coord, ref wall4);
        Rotate(true, Mathf.Atan2(wall4[2].coord.y, wall4[2].coord.x), ref wall4);
        CreateWalls(ref wall4);

        // cameraTarget 설정
        cameraTarget.transform.position = new Vector3(wall4[1].coord.x / 2f, 0, wall4[1].coord.y / 2f);
        Camera.main.transform.position = cameraTarget.transform.position + new Vector3(0, wall4[1].coord.x, -wall4[1].coord.y / 1.4f);

        SetPointsToExcludeForObject();
        List<List<int>> objectGroup = SetObjectGroups();

        foreach (var objectIdx in objectGroup)
        {
            List<int> idices = objectIdx;
            SetPointsToExclude(ref idices);
            List<int> objtHull = ConvexHull(ref idices);
            List<Point> objt4 = MergeHullTo4Points(false, ref objtHull);

            SortObjtPoints(ref objt4);

            Vector2 transO = objt4[3].coord;
            Translate(false, transO, ref objt4);
            Vector2 rotO = objt4[2].coord;
            Rotate(false, Mathf.Atan2(rotO.y, rotO.x), ref objt4);

            CreateObjt(ref objt4, rotO, transO);
        }
    }
}
