package object;

import Main.GamePanel;
import entity.Entity;

public class OBJ_Potion_Red extends Entity {

    GamePanel gp;
    public static final String objName = "Red Potion";

    public OBJ_Potion_Red(GamePanel gp) {
        super(gp);

        this.gp = gp;

        type = type_consumable;
        name = objName;
        value = 5;
        down1 = setup("/objects/potion_red", gp.tileSize, gp.tileSize);
        description = "[" + name + "]\nHeals your life by " + value + ".";
        price = 75;
        stackable = true;
    }

    public boolean use(Entity entity) {

        gp.gameState = gp.dialogueState;
        gp.ui.currentDialogue = "You drink the " + name + "!\nYour life has been recovered by " + value + ".";
        entity.life += value;
        gp.playSE(2);
        return true;
    }
}
