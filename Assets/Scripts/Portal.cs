using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Portal : MonoBehaviour {

    public Vector2 spawnOffset;

    EnemySpawner enemySpawner;

    public void SetEnemySpawner(EnemySpawner es) {
        enemySpawner = es;
    }

    // Called from portal_close animation event
    void DeletePortal() {
        Destroy(gameObject);
    }

    void SpawnEnemy() {
        enemySpawner.SpawnEnemy(transform.position + new Vector3(spawnOffset.x, spawnOffset.y, 0));
    }

}
