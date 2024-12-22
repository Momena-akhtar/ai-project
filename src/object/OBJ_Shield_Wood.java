package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_Shield_Wood extends Entity {

    public static final String objName = "Wood Shield";

    public OBJ_Shield_Wood(GamePanel gp) {
        super(gp);

        type = type_shield;
        name = objName;
        down1 = setup("/objects/shield_wood", gp.tileSize, gp.tileSize);
        defenseValue = 1;
        description = "[" + name + "]\nMade by wood.";
        price = 30;
        knockBackPower = 3;

        guardUp = setup("/player/boy_guard_up", gp.tileSize, gp.tileSize);
        guardDown = setup("/player/boy_guard_down", gp.tileSize, gp.tileSize);
        guardRight = setup("/player/boy_guard_right", gp.tileSize, gp.tileSize);
        guardLeft = setup("/player/boy_guard_left", gp.tileSize, gp.tileSize);
    }

}
