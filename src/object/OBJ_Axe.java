package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_Axe extends Entity {

    public static final String objName = "Woodcutter's Axe";// name stored in static variable to ease in saving loading
                                                            // gme and to avoid typos

    public OBJ_Axe(GamePanel gp) {
        super(gp);

        type = type_axe;
        name = objName;
        down1 = setup("/objects/axe", gp.tileSize, gp.tileSize);
        attackValue = 2;
        attackArea.width = 30;
        attackArea.height = 30;
        description = "[" + name + "]\nA bit rusty but still can \ncut some trees.";
        price = 75;
        knockBackPower = 10;
        motion1_duration = 20;
        motion2_duration = 40;
    }
}