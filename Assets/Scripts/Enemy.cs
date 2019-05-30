using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour {

    float movementSpeed;
    Vector3 movementDirection = new Vector3(0, -1, 0);
    Renderer rend;
    Rigidbody2D rigid;

    void Start() {
        rend = GetComponent<Renderer>();    
        rigid = GetComponent<Rigidbody2D>();
    }

    public void SetMovementSpeed(float ms) {
        movementSpeed = ms;
    }

    // Update is called once per frame
    void FixedUpdate() {
        //    transform.Translate(movementSpeed * movementDirection);
        rigid.velocity = movementDirection * movementSpeed;
        if (!rend.isVisible || transform.position.y <= -10) {
            Destroy(gameObject);
        }
    }
}
