using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class SceneChanger : MonoBehaviour
{
    public TMP_InputField fileName;
    public TMP_InputField wallThickness;
    public TMP_InputField scaleConstant;

    public void Save()
    {
        PlayerPrefs.SetString("fileName", fileName.text);
        PlayerPrefs.SetFloat("wallThickness", float.Parse(wallThickness.text));
        PlayerPrefs.SetInt("scaleConstant", int.Parse(scaleConstant.text));
    }

    public void GoToMain()
    {
        SceneManager.LoadScene("Scenes/MainScene");
    }

    public void GoToStart()
    {
        SceneManager.LoadScene("Scenes/StartScene");
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
            Application.Quit();

        if (Input.GetKeyDown(KeyCode.Return))
        {
            Save();
            GoToMain();
        }
    }
}
