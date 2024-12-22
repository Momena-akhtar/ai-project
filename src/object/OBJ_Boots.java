package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_Boots extends Entity {
    public static final String objName = "Boots";

    GamePanel gp;

    public OBJ_Boots(GamePanel gp) {
        super(gp);
        this.gp = gp;

        type = type_consumable;
        name = objName;

        down1 = setup("/objects/boots", gp.tileSize, gp.tileSize);
        description = "[" + name + "]\nNew shiny boots,\nincreases your speed.";
        price = 100;
        stackable = true;
    }

    public boolean use(Entity entity) {

        gp.gameState = gp.dialogueState;
        gp.ui.currentDialogue = "You wear " + name + "!\nYour speed has been increased for some time.";
        gp.playSE(2);
        gp.player.speed += 2;
        return true;
    }
}
