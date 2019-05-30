using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Portal : MonoBehaviour {
    
    // Called from portal_close animation event
    void DeletePortal() {
        Destroy(gameObject);
    }

}
