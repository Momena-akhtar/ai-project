package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_BadBoots extends Entity {
    public static final String objName = "Bad Boots";

    GamePanel gp;

    public OBJ_BadBoots(GamePanel gp) {
        super(gp);
        this.gp = gp;

        type = type_consumable;
        name = objName;

        down1 = setup("/objects/bad boots", gp.tileSize, gp.tileSize);
        description = "[" + name + "]\nCursed boots,\ndecreases your speed.";
        price = 25;
        stackable = true;
    }

    public boolean use(Entity entity) {

        gp.gameState = gp.dialogueState;
        gp.ui.currentDialogue = "You wear " + name + "!\nYour speed has been decreased for some time.";
        gp.playSE(3);
        gp.player.speed -= 2;
        return true;
    }
}