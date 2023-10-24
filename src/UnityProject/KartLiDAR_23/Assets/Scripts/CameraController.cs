using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public GameObject target;
    public int cameraBoundary = 50;

    public float rotSpeed = 100f;
    public float verticalSpeed = 10f;
    public float zoomSpeed = 5000;

    void Update()
    {
        // 카메라 회전
        transform.RotateAround(target.transform.position, target.transform.up, -Input.GetAxis("Horizontal") * Time.deltaTime * rotSpeed);

        // 카메라 수직 이동
        Vector3 verticalMovement = new Vector3(0, Input.GetAxis("Vertical") * Time.deltaTime * verticalSpeed, 0);

        // 카메라 확대 축소
        Vector3 zoomMovement = transform.localRotation * Vector3.forward * Input.GetAxis("Mouse ScrollWheel") * Time.deltaTime * zoomSpeed;

        transform.position += verticalMovement + zoomMovement;
        transform.position = new Vector3(Mathf.Clamp(transform.position.x, -cameraBoundary, cameraBoundary),
                                         Mathf.Clamp(transform.position.y, 0, cameraBoundary),
                                         Mathf.Clamp(transform.position.z, -cameraBoundary, cameraBoundary));

        transform.LookAt(target.transform.position);
    }
}
