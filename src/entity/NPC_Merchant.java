package entity;

import Main.GamePanel;
import object.OBJ_Axe;
import object.OBJ_BadBoots;
import object.OBJ_Boots;
import object.OBJ_Lantern;
import object.OBJ_Potion_Red;
import object.OBJ_Shield_Blue;
import object.OBJ_Sword_Normal;
import object.OBJ_Tent;

public class NPC_Merchant extends Entity {

    public NPC_Merchant(GamePanel gp) {

        super(gp);

        direction = "down";
        speed = 1;

        getImage();
        setDialogue();
        setItems();
    }

    public void getImage() {

        up1 = setup("/npc/merchant_down_1", gp.tileSize, gp.tileSize);
        up2 = setup("/npc/merchant_down_2", gp.tileSize, gp.tileSize);
        down1 = setup("/npc/merchant_down_1", gp.tileSize, gp.tileSize);
        down2 = setup("/npc/merchant_down_2", gp.tileSize, gp.tileSize);
        left1 = setup("/npc/merchant_down_1", gp.tileSize, gp.tileSize);
        left2 = setup("/npc/merchant_down_2", gp.tileSize, gp.tileSize);
        right1 = setup("/npc/merchant_down_1", gp.tileSize, gp.tileSize);
        right2 = setup("/npc/merchant_down_2", gp.tileSize, gp.tileSize);

    }

    public void setDialogue() {
        dialogues[0] = "Wana trade some of ya shiny things?\nhehe";
    }

    public void setItems() {

        inventory.add(new OBJ_Potion_Red(gp));
        inventory.add(new OBJ_Sword_Normal(gp));
        inventory.add(new OBJ_Shield_Blue(gp));
        inventory.add(new OBJ_BadBoots(gp));
        inventory.add(new OBJ_Boots(gp));
        inventory.add(new OBJ_Lantern(gp));
        inventory.add(new OBJ_Tent(gp));
        inventory.add(new OBJ_Axe(gp));
        inventory.add(new OBJ_Shield_Blue(gp));
    }

    public void speak() {

        super.speak();

        gp.gameState = gp.tradeState;
        gp.ui.npc = this;
    }
}
