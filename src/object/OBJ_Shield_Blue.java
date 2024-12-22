package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_Shield_Blue extends Entity {

    public static final String objName = "Blue Shield";

    public OBJ_Shield_Blue(GamePanel gp) {
        super(gp);

        type = type_shield;
        name = objName;
        down1 = setup("/objects/shield_blue", gp.tileSize, gp.tileSize);
        defenseValue = 2;
        description = "[" + name + "]\nA shiny blue shield.";
        price = 250;
        knockBackPower = 6;

        guardUp = setup("/player/boy_guard_up_blue", gp.tileSize, gp.tileSize);
        guardDown = setup("/player/boy_guard_down_blue", gp.tileSize, gp.tileSize);
        guardRight = setup("/player/boy_guard_right_blue", gp.tileSize, gp.tileSize);
        guardLeft = setup("/player/boy_guard_left_blue", gp.tileSize, gp.tileSize);
    }
}
