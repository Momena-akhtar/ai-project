package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_Key extends Entity {

    GamePanel gp;
    public static final String objName = "Key";

    public OBJ_Key(GamePanel gp) {
        super(gp);
        this.gp = gp;

        type = type_consumable;
        name = objName;
        down1 = setup("/objects/key", gp.tileSize, gp.tileSize);
        description = "[" + name + "]\nIt unlocks doors.";
        stackable = true;
    }

    public boolean use(Entity entity) {

        gp.gameState = gp.dialogueState;

        int objIndex = getDetected(entity, gp.obj, "Door");
        if (objIndex != 999) {
            gp.ui.currentDialogue = "You use the " + name + " and open the door";
            gp.playSE(4);
            gp.obj[gp.currentMap][objIndex] = null;
            return true;

        } else {
            gp.ui.currentDialogue = "Oh, brilliant idea! Trying to unlock thin air now, are \nwe?";
            return false;
        }

    }
}