using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour {

    float movementSpeed;
    Vector3 movementDirection = new Vector3(0, -1, 0);
    Renderer rend;

    void Start() {
        rend = GetComponent<Renderer>();    
    }

    public void SetMovementSpeed(float ms) {
        movementSpeed = ms;
    }

    // Update is called once per frame
    void FixedUpdate() {
        transform.Translate(movementSpeed * movementDirection);
        if (!rend.isVisible || transform.position.y <= -10) {
            Destroy(gameObject);
        }
    }
}
