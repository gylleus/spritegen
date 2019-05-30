using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemySpawner : MonoBehaviour {

    public float initialEnemyMovementSpeed;
    // How much the movement speed of enemies will scale with difficulty
    public float movementSpeedDifficultyScaling = 0f;

    public GameObject enemyTemplate;

    GameController gameController;

    void Start() {
        gameController = GetComponent<GameController>();
    }

    public void SpawnEnemy(Vector3 position) {
        GameObject newEnemy = Instantiate(enemyTemplate, position, Quaternion.identity);
        float enemyMS = initialEnemyMovementSpeed + gameController.GetDifficulty() * movementSpeedDifficultyScaling;
        newEnemy.GetComponent<Enemy>().SetMovementSpeed(enemyMS);
    }

    
}
