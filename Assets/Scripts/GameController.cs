using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameController : MonoBehaviour {

    public float initialPortalsPerSecond = 1.0f;
    public float startDifficulty = 1.0f;

    public float difficultyIncreaseFactor = 1.0f;
    public float difficultyIncreaseAddend = 0.0f;
    public float difficultyIncreaseInterval = 10.0f;
    // Padding between edges and portals spawned
    public Vector2 portalEdgePadding;
    public float portalMinSpaceBetween;


    public GameObject portal;

    Vector2[] portalSpawnAreas;
    float currentDifficulty;
    float portalsPerSecond;

    // Start is called before the first frame update
    void Start() {
        portalSpawnAreas = SetSpawnAreas();
        currentDifficulty = startDifficulty;
        portalsPerSecond = initialPortalsPerSecond;

        StartCoroutine("PortalSpawner");
        StartCoroutine("DifficultyIncreaser");
    }

    IEnumerator DifficultyIncreaser() {
        while (true) {
            UpdateDifficulty();
            yield return new WaitForSeconds(difficultyIncreaseFactor);
        }
    }

    IEnumerator PortalSpawner() {
        while (true) {
            Vector2 spawnLocation = SelectSpawnArea();
            Instantiate(portal, new Vector3(spawnLocation.x, spawnLocation.y), Quaternion.identity);
            yield return new WaitForSeconds((1 / portalsPerSecond));
        }
    }

    void UpdateDifficulty() {
        currentDifficulty =  currentDifficulty * difficultyIncreaseFactor + difficultyIncreaseAddend;
        portalsPerSecond = initialPortalsPerSecond * currentDifficulty;
    }

    Vector2 SelectSpawnArea() {
        return portalSpawnAreas[Random.Range(0, portalSpawnAreas.Length - 1)];
    }

    Vector2[] SetSpawnAreas() {
        float ySpawnPos = Screen.height - portalEdgePadding.y;
        float cameraZ = Camera.main.transform.position.z;
        // 0,0 is bottom left - Screen.width, Screen.heigh is top right
        Vector3 leftPoint = Camera.main.ScreenToWorldPoint(new Vector3(portalEdgePadding.x, ySpawnPos, cameraZ));
        Vector3 rightPoint = Camera.main.ScreenToWorldPoint(new Vector3(Screen.width - portalEdgePadding.x, ySpawnPos, cameraZ));

        float xDif = rightPoint.x - leftPoint.x;
        SpriteRenderer sr = portal.GetComponent<SpriteRenderer>();
        float portalXWidth = portal.GetComponent<SpriteRenderer>().size.x + portalMinSpaceBetween;
        int portalsAmount = Mathf.FloorToInt(xDif / portalXWidth);
        Vector2[] spawnAreas = new Vector2[portalsAmount];

        // Calculate how much of xDif we don't use so we can move portals accordingly
        float xRest = xDif - portalXWidth * portalsAmount;
        // Pad by xRest/2 on left and right side
        float xSpawnBegin = portalEdgePadding.x + xRest/2;

        for (int i = 0; i < portalsAmount; i++) {
            spawnAreas[i] = new Vector2(leftPoint.x + xSpawnBegin + i * portalXWidth + portal.GetComponent<SpriteRenderer>().size.x/2, leftPoint.y);
        }
        return spawnAreas;
    }

    // Update is called once per frame
    void Update() {
        
    }
}
