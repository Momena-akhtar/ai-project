package Main;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontFormatException;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import entity.Entity;
import object.OBJ_Coin_Bronze;
import object.OBJ_Heart;
import object.OBJ_ManaCrystal;

public class UI {
    GamePanel gp;
    Graphics2D g2;
    public Font maruMonica, purisaB;
    BufferedImage heart_full, heart_half, heart_blank, crystal_full, crystal_blank, coin;
    public boolean messageOn = false;
    ArrayList<String> message = new ArrayList<>();
    ArrayList<Integer> messageCounter = new ArrayList<>();
    public boolean gameFinished = false;
    public String currentDialogue = "";
    public int commandNum = 0;
    public int titleScreenState = 0;// 0= first screen, 1= second screen
    public int playerSlotCol = 0;
    public int playerSlotrow = 0;
    public int npcSlotCol = 0;
    public int npcSlotrow = 0;
    int subState = 0;
    int counter = 0;
    public Entity npc;

    public UI(GamePanel gp) {
        this.gp = gp;

        try {
            InputStream is = getClass().getResourceAsStream("/font/x12y16pxMaruMonica.ttf");
            maruMonica = Font.createFont(Font.TRUETYPE_FONT, is);

            is = getClass().getResourceAsStream("/font/Purisa Bold.ttf");
            purisaB = Font.createFont(Font.TRUETYPE_FONT, is);
        } catch (FontFormatException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // CREATE HUD OBJECT
        Entity heart = new OBJ_Heart(gp);
        heart_full = heart.image;
        heart_half = heart.image2;
        heart_blank = heart.image3;
        Entity crystal = new OBJ_ManaCrystal(gp);
        crystal_full = crystal.image;
        crystal_blank = crystal.image2;
        Entity bCoin = new OBJ_Coin_Bronze(gp);
        coin = bCoin.down1;
    }

    public void addMessage(String text) {

        message.add(text);
        messageCounter.add(0);
    }

    public void draw(Graphics2D g2) {
        this.g2 = g2;

        // g2.setFont(purisaB);
        g2.setFont(maruMonica);
        // g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
        // RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2.setColor(Color.white);

        // TITLE STATE
        if (gp.gameState == gp.titleState) {
            drawTitleScreen();
        }
        // PLAY STATE
        if (gp.gameState == gp.playState) {
            drawPlayerLife();
            drawMessage();
        }

        // PAUSE STATE
        if (gp.gameState == gp.pauseState) {
            drawPlayerLife();
            drawPauseScreen();
        }

        // DIALOGUE STATE
        if (gp.gameState == gp.dialogueState) {
            drawDialogueScreen();
        }

        // CHARACTER STATE
        if (gp.gameState == gp.characterState) {
            drawCharacterScreen();
            drawInventory(gp.player, true);
        }

        // OPTIONS STATE
        if (gp.gameState == gp.optionsState) {
            drawOptionsScreen();
        }

        // GAME OVER STATE
        if (gp.gameState == gp.gameOverState) {
            drawGameOverScreen();
        }

        // TRANSITION STATE
        if (gp.gameState == gp.transitionState) {
            drawTransitionState();
        }

        // TRADE STATE
        if (gp.gameState == gp.tradeState) {
            drawTadeScreen();
        }

        // TRADE STATE
        if (gp.gameState == gp.sleepState) {
            drawSleepScreen();
        }

    }

    public void drawPlayerLife() {

        int x = gp.tileSize / 2;
        int y = gp.tileSize / 3;
        int i = 0;

        // DRAW MAX LIFE
        while (i < gp.player.maxLife / 2) {
            g2.drawImage(heart_blank, x, y, null);
            i++;
            x += gp.tileSize;
        }

        // RESET
        x = gp.tileSize / 2;
        y = gp.tileSize / 3;
        i = 0;

        // DRAW CURRENT LIFE
        while (i < gp.player.life) {
            g2.drawImage(heart_half, x, y, null);
            i++;
            if (i < gp.player.life) {
                g2.drawImage(heart_full, x, y, null);
                i++;
                x += gp.tileSize;
            }
        }

        // DRAW MAX MANA
        x = (int) (gp.tileSize / 2.35);
        y = (gp.tileSize * 4) / 3;
        i = 0;
        while (i < gp.player.maxMana) {
            g2.drawImage(crystal_blank, x, y, null);
            i++;
            x += (2 * gp.tileSize) / 3;
        }

        // RESET
        x = (int) (gp.tileSize / 2.35);
        y = (gp.tileSize * 4) / 3;
        i = 0;

        // DRAW CURRENT MANA
        while (i < gp.player.mana) {
            g2.drawImage(crystal_full, x, y, null);
            i++;
            x += (2 * gp.tileSize) / 3;
        }
    }

    public void drawMessage() {

        int messageX = gp.tileSize;
        int messageY = gp.tileSize * 4;
        g2.setFont(g2.getFont().deriveFont(Font.BOLD, 32F));

        for (int i = 0; i < message.size(); i++) {

            if (message.get(i) != null) {

                g2.setColor(Color.black);
                g2.drawString(message.get(i), messageX + 2, messageY + 2);
                g2.setColor(Color.white);
                g2.drawString(message.get(i), messageX, messageY);

                int counter = messageCounter.get(i) + 1; // message counter++
                messageCounter.set(i, counter);

                messageY += 50;

                if (messageCounter.get(i) > 180) {
                    message.remove(i);
                    messageCounter.remove(i);
                }
            }
        }
    }

    public void drawTitleScreen() {

        if (titleScreenState == 0) {
            g2.setColor(new Color(0, 0, 0));
            g2.fillRect(0, 0, gp.screenWidth, gp.screenHeight);

            // TITLE NAME
            g2.setFont(g2.getFont().deriveFont(Font.BOLD, 120));
            String text = "ADVENTURE TIME";
            int x = getXforCenteredText(text);
            int y = gp.tileSize * 3;

            // SHADOW
            g2.setColor(Color.gray);
            g2.drawString(text, x + 5, y + 5);

            // MAIN COLOR
            g2.setColor(Color.white);
            g2.drawString(text, x, y);

            // CHARACTER IMAGE
            x = gp.screenWidth / 2 - (gp.tileSize * 2) / 2;
            y += gp.tileSize + 2;
            g2.drawImage(gp.player.down1, x, y, gp.tileSize * 2, gp.tileSize * 2, null);

            // MENU
            g2.setFont(g2.getFont().deriveFont(Font.BOLD, 48F));

            text = "NEW GAME";
            x = getXforCenteredText(text);
            y += gp.tileSize * 4;
            g2.drawString(text, x, y);
            if (commandNum == 0) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }

            text = "LOAD GAME";
            x = getXforCenteredText(text);
            y += gp.tileSize;
            g2.drawString(text, x, y);
            if (commandNum == 1) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }

            text = "QUIT";
            x = getXforCenteredText(text);
            y += gp.tileSize;
            g2.drawString(text, x, y);
            if (commandNum == 2) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }
        } else if (titleScreenState == 1) {

            // CLASS SELECTION SCREEN
            g2.setColor(Color.white);
            g2.setFont(g2.getFont().deriveFont(42F));

            String text = "Select Your Class!";
            int x = getXforCenteredText(text);
            int y = gp.tileSize * 3;
            g2.drawString(text, x, y);

            text = "Fighter!";
            x = getXforCenteredText(text);
            y += gp.tileSize * 3;
            g2.drawString(text, x, y);
            if (commandNum == 0) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }

            text = "Thief!";
            x = getXforCenteredText(text);
            y += gp.tileSize;
            g2.drawString(text, x, y);
            if (commandNum == 1) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }

            text = "Sorcerer!";
            x = getXforCenteredText(text);
            y += gp.tileSize;
            g2.drawString(text, x, y);
            if (commandNum == 2) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }

            text = "Back";
            x = getXforCenteredText(text);
            y += gp.tileSize * 2;
            g2.drawString(text, x, y);
            if (commandNum == 3) {
                g2.drawString(">", x - gp.tileSize / 2, y);
            }
        }
    }

    public void drawPauseScreen() {

        g2.setFont(g2.getFont().deriveFont(Font.PLAIN, 80));
        String text = "PAUSED";
        int x = getXforCenteredText(text);
        int y = gp.screenHeight / 2;
        g2.drawString(text, x, y);
    }

    public void drawDialogueScreen() {

        // WINDOW
        int x = gp.tileSize * 3;
        int y = gp.tileSize / 2;
        int width = gp.screenWidth - (gp.tileSize * 6);
        int height = gp.tileSize * 4;
        drawSubWindow(x, y, width, height);

        g2.setFont(g2.getFont().deriveFont(Font.PLAIN, 32));
        x += gp.tileSize;
        y += gp.tileSize;

        for (String line : currentDialogue.split("\n")) {
            g2.drawString(line, x, y);
            y += 40;
        }
    }

    public void drawCharacterScreen() {

        // CREATE A FRAME
        final int frameX = gp.tileSize * 2;
        final int frameY = gp.tileSize;
        final int frameWidth = gp.tileSize * 5;
        final int frameHeight = gp.tileSize * 10;

        drawSubWindow(frameX, frameY, frameWidth, frameHeight);

        // TEXT
        g2.setColor(Color.white);
        g2.setFont(g2.getFont().deriveFont(32F));

        int textX = frameX + 20;
        int textY = frameY + gp.tileSize;
        final int lineHeight = 35;

        // NAMES
        g2.drawString("Level", textX, textY);
        textY += lineHeight;
        g2.drawString("Life", textX, textY);
        textY += lineHeight;
        g2.drawString("Mana", textX, textY);
        textY += lineHeight;
        g2.drawString("Strenght", textX, textY);
        textY += lineHeight;
        g2.drawString("Dexterity", textX, textY);
        textY += lineHeight;
        g2.drawString("Attack", textX, textY);
        textY += lineHeight;
        g2.drawString("Defense", textX, textY);
        textY += lineHeight;
        g2.drawString("Exp", textX, textY);
        textY += lineHeight;
        g2.drawString("Next Level", textX, textY);
        textY += lineHeight;
        g2.drawString("Coin", textX, textY);
        textY += lineHeight + 10;
        g2.drawString("Weapon", textX, textY);
        textY += lineHeight + 15;
        g2.drawString("Shield", textX, textY);
        textY += lineHeight;

        // VALUES
        int tailX = (frameX + frameWidth) - 30;
        // Reset textY
        textY = frameY + gp.tileSize;
        String value;

        value = String.valueOf(gp.player.level);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.life + "/" + gp.player.maxLife);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.mana + "/" + gp.player.maxMana);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.strenght);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.dexterity);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.attack);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.defense);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.exp);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.nextLevelExp);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        value = String.valueOf(gp.player.coin);
        textX = getXforAlignToRightText(value, tailX);
        g2.drawString(value, textX, textY);
        textY += lineHeight;

        g2.drawImage(gp.player.currentWeapon.down1, tailX - gp.tileSize, textY - 24, null);
        textY += gp.tileSize;

        g2.drawImage(gp.player.currentShield.down1, tailX - gp.tileSize, textY - 24, null);

    }

    public void drawInventory(Entity entity, boolean cursor) {

        int frameX = 0;
        int frameY = 0;
        int frameWidth = 0;
        int frameHeight = 0;
        int slotCol = 0;
        int slotrow = 0;

        if (entity == gp.player) {
            frameX = gp.tileSize * 12;
            frameY = gp.tileSize;
            frameWidth = gp.tileSize * 6;
            frameHeight = gp.tileSize * 5;
            slotCol = playerSlotCol;
            slotrow = playerSlotrow;
        } else {
            frameX = gp.tileSize * 2;
            frameY = gp.tileSize;
            frameWidth = gp.tileSize * 6;
            frameHeight = gp.tileSize * 5;
            slotCol = npcSlotCol;
            slotrow = npcSlotrow;
        }

        // FRAME
        drawSubWindow(frameX, frameY, frameWidth, frameHeight);

        // SLOT
        final int slotXstart = frameX + 20;
        final int slotYstart = frameY + 20;
        int slotX = slotXstart;
        int slotY = slotYstart;
        int slotSize = gp.tileSize + 3;

        // DRAW PLAYER'S ITEMS
        for (int i = 0; i < entity.inventory.size(); i++) {

            // EQUIP CURSOR
            if (entity.inventory.get(i) == entity.currentWeapon ||
                    entity.inventory.get(i) == entity.currentShield || entity.inventory.get(i) == entity.currentLight) {

                g2.setColor(new Color(240, 190, 90));
                g2.fillRoundRect(slotX, slotY, gp.tileSize, gp.tileSize, 10, 10);

            }

            g2.drawImage(entity.inventory.get(i).down1, slotX, slotY, null);

            // DISPLAY AMOUNT
            if (entity == gp.player && entity.inventory.get(i).amount > 1) {

                g2.setFont(g2.getFont().deriveFont(32f));
                int amountX;
                int amountY;

                String s = "" + entity.inventory.get(i).amount;
                amountX = getXforAlignToRightText(s, slotX + 44);
                amountY = slotY + gp.tileSize;

                // SHADOW
                g2.setColor(new Color(60, 60, 60));
                g2.drawString(s, amountX, amountY);
                // NUMBER
                g2.setColor(Color.white);
                g2.drawString(s, amountX - 3, amountY - 3);
            }
            slotX += slotSize;

            if (i == 4 || i == 9 || i == 14) {
                slotX = slotXstart;
                slotY += slotSize;
            }
        }

        // CURSOR
        if (cursor == true) {
            int cursorX = slotXstart + (slotSize * slotCol);
            int cursorY = slotYstart + (slotSize * slotrow);
            int cursorWidth = gp.tileSize;
            int cursorHeight = gp.tileSize;

            // DRAW CURSOR
            g2.setColor(Color.white);
            g2.setStroke(new BasicStroke(3));
            g2.drawRoundRect(cursorX, cursorY, cursorWidth, cursorHeight, 10, 10);

            // DESCRIPTION FRAME
            int dFrameX = frameX;
            int dFrameY = frameY + frameHeight;
            int dFrameWidth = frameWidth;
            int dFrameHeight = gp.tileSize * 3;

            // DRAW DESCRIPTION TEXT
            int textX = dFrameX + 20;
            int textY = dFrameY + gp.tileSize;
            g2.setFont(g2.getFont().deriveFont(28F));

            int itemIndex = getItemIdexOnSlot(slotCol, slotrow);
            if (itemIndex < entity.inventory.size() && entity.inventory.get(itemIndex).description != null) {

                drawSubWindow(dFrameX, dFrameY, dFrameWidth, dFrameHeight);

                for (String line : entity.inventory.get(itemIndex).description.split("\n")) {

                    g2.drawString(line, textX, textY);
                    textY += 32;

                }
            }
        }
    }

    public int getItemIdexOnSlot(int slotCol, int slotrow) {
        int itemIndex = slotCol + (slotrow * 5);
        return itemIndex;
    }

    public void drawGameOverScreen() {
        g2.setColor(new Color(0, 0, 0, 150));
        g2.fillRect(0, 0, gp.screenWidth, gp.screenHeight);

        int textX;
        int textY;
        String text = "Game Over";
        g2.setFont(g2.getFont().deriveFont(Font.BOLD, 110f));

        // SHADOW
        g2.setColor(Color.black);
        textX = getXforCenteredText(text);
        textY = gp.tileSize * 4;
        g2.drawString(text, textX, textY);

        // ACTUAL
        g2.setColor(Color.white);
        g2.drawString(text, textX - 4, textY - 4);

        // RETRY
        g2.setFont(g2.getFont().deriveFont(50f));
        text = "Retry";
        textX = getXforCenteredText(text);
        textY += gp.tileSize * 4;
        g2.drawString(text, textX, textY);
        if (commandNum == 0) {
            g2.drawString(">", textX - 40, textY);
        }

        // Back to the title screen
        text = "Quit";
        textX = getXforCenteredText(text);
        textY += 55;
        g2.drawString(text, textX, textY);
        if (commandNum == 1) {
            g2.drawString(">", textX - 40, textY);
        }

    }

    public void drawOptionsScreen() {

        g2.setColor(Color.white);
        g2.setFont(g2.getFont().deriveFont(32F));

        // SUB WINDOW
        int frameX = gp.tileSize * 6;
        int frameY = gp.tileSize;
        int frameWidth = gp.tileSize * 8;
        int frameHeight = gp.tileSize * 10;
        drawSubWindow(frameX, frameY, frameWidth, frameHeight);

        switch (subState) {
            case 0:
                options_top(frameX, frameY);
                break;
            case 1:
                options_controls(frameX, frameY);
                break;
            case 2:
                options_endGameConfirmation(frameX, frameY);
                break;
        }

        gp.keyH.enterPressed = false;
    }

    public void options_top(int frameX, int frameY) {

        int textX;
        int textY;

        // TITLE
        String text = "Options";
        textX = getXforCenteredText(text);
        textY = frameY + gp.tileSize;
        g2.drawString(text, textX, textY);

        // FULL SCREEN ON/OFF
        textX = frameX + gp.tileSize;
        textY += gp.tileSize * 2;
        g2.drawString("Full Screen", textX, textY);
        if (commandNum == 0) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed == true) {
                if (gp.fullScreen == false) {
                    gp.setFullScreen();
                } else if (gp.fullScreen == true) {
                    gp.setDefaultScreen();
                }

            }
        }

        // MUSIC
        textY += gp.tileSize;
        g2.drawString("MUSIC", textX, textY);
        if (commandNum == 1) {
            g2.drawString(">", textX - 25, textY);
        }

        // SE
        textY += gp.tileSize;
        g2.drawString("SE", textX, textY);
        if (commandNum == 2) {
            g2.drawString(">", textX - 25, textY);
        }

        // CONTROLS
        textY += gp.tileSize;
        g2.drawString("CONTROLS", textX, textY);
        if (commandNum == 3) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed == true) {
                subState = 1;
                commandNum = 0;
            }
        }

        // END GAME
        textY += gp.tileSize;
        g2.drawString("END GAME", textX, textY);
        if (commandNum == 4) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed) {
                subState = 2;
                commandNum = 0;
            }
        }

        // BACK
        textY += gp.tileSize * 2;
        g2.drawString("Back", textX, textY);
        if (commandNum == 5) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed) {
                gp.gameState = gp.playState;
                commandNum = 0;
            }
        }

        // FULL SCREEN CHECK BOX
        textX = frameX + (int) (gp.tileSize * 4.5);
        textY = frameY + (int) (gp.tileSize * 2.5);
        g2.setStroke(new BasicStroke(3));
        g2.drawRect(textX, textY, gp.tileSize / 2, gp.tileSize / 2);
        if (gp.fullScreen) {
            g2.fillRect(textX, textY, gp.tileSize / 2, gp.tileSize / 2);
        }

        // MUSIC
        textY += gp.tileSize;
        g2.drawRect(textX, textY, (int) (gp.tileSize * 2.5), gp.tileSize / 2);
        int volumeWidth = (gp.tileSize / 2) * gp.music.volumeScale;
        g2.fillRect(textX, textY, volumeWidth, gp.tileSize / 2);

        // SE
        textY += gp.tileSize;
        g2.drawRect(textX, textY, (int) (gp.tileSize * 2.5), gp.tileSize / 2);
        volumeWidth = (gp.tileSize / 2) * gp.se.volumeScale;
        g2.fillRect(textX, textY, volumeWidth, gp.tileSize / 2);
    }

    public void options_controls(int frameX, int frameY) {

        int textX;
        int textY;
        int padding = (int) (gp.tileSize / 1.3);

        // TITLE
        String text = "Controls";
        textX = getXforCenteredText(text);
        textY = frameY + gp.tileSize;
        g2.drawString(text, textX, textY);

        textX = frameX + gp.tileSize;
        textY += gp.tileSize;

        g2.setFont(purisaB);
        g2.setFont(g2.getFont().deriveFont(20f));

        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
                RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        g2.drawString("Move", textX, textY);
        textY += padding;
        g2.drawString("Confirm/Attack", textX, textY);
        textY += padding;
        g2.drawString("Guard/Parry", textX, textY);
        textY += padding;
        g2.drawString("Shoot/Cast", textX, textY);
        textY += padding;
        g2.drawString("Map/Mini Map", textX, textY);
        textY += padding;
        g2.drawString("Character Screen", textX, textY);
        textY += padding;
        g2.drawString("Pause", textX, textY);
        textY += padding;
        g2.drawString("Options", textX, textY);
        textY += padding;
        g2.drawString("Full Screen", textX, textY);
        textY += padding;

        textX = frameX + gp.tileSize * 3;
        textY = frameY + gp.tileSize * 2;
        g2.drawString("WASD/Arrow Keys", textX, textY);
        textX = frameX + gp.tileSize * 6;
        textY += padding;
        g2.drawString("ENTER", textX, textY);
        textY += padding;
        g2.drawString("B", textX, textY);
        textY += padding;
        g2.drawString("SPACE", textX, textY);
        textY += padding;
        g2.drawString("M/X", textX, textY);
        textY += padding;
        g2.drawString("C", textX, textY);
        textY += padding;
        g2.drawString("P", textX, textY);
        textY += padding;
        g2.drawString("ESC", textX, textY);
        textY += padding;
        g2.drawString("F", textX, textY);
        textY += padding;

        // BACK
        textX = frameX + gp.tileSize;
        textY = frameY + gp.tileSize * 9;
        g2.setFont(maruMonica);
        g2.setFont(g2.getFont().deriveFont(32f));
        g2.drawString("Back", textX, textY);
        if (commandNum == 0) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed) {
                subState = 0;
                commandNum = 3;
            }
        }
    }

    public void options_endGameConfirmation(int frameX, int frameY) {

        int textX;
        int textY;

        // TITLE
        String text = "Quit the game and return";
        textX = getXforCenteredText(text);
        textY = frameY + gp.tileSize * 2;
        g2.drawString(text, textX, textY);
        text = "to the main menu?";
        textX = getXforCenteredText(text);
        textY += gp.tileSize;
        g2.drawString(text, textX, textY);

        // YES

        text = "Yes";
        textX = getXforCenteredText(text);
        textY += gp.tileSize * 3;
        g2.drawString(text, textX, textY);
        if (commandNum == 0) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed) {
                subState = 0;
                gp.stopMusic();
                gp.resetGame(true);
                gp.gameState = gp.titleState;
            }
        }

        // NO
        text = "No";
        textX = getXforCenteredText(text);
        textY += gp.tileSize;
        g2.drawString(text, textX, textY);
        if (commandNum == 1) {
            g2.drawString(">", textX - 25, textY);
            if (gp.keyH.enterPressed) {
                subState = 0;
                commandNum = 4;
            }
        }
    }

    public void drawTransitionState() {

        counter++;
        g2.setColor(new Color(0, 0, 0, counter * 5));
        g2.fillRect(0, 0, gp.screenWidth, gp.screenHeight);

        if (counter == 50) {
            counter = 0;
            gp.gameState = gp.playState;
            gp.currentMap = gp.eHandler.tempMap;
            gp.player.worldX = gp.eHandler.tempCol * gp.tileSize;
            gp.player.worldY = gp.eHandler.tempRow * gp.tileSize;
            gp.eHandler.previousEventX = gp.player.worldX;
            gp.eHandler.previousEventY = gp.player.worldY;
            gp.eHandler.canTouchEvent = false;
        }
    }

    public void drawTadeScreen() {

        switch (subState) {
            case 0:
                trade_select();
                break;
            case 1:
                trade_buy();
                break;
            case 2:
                trade_sell();
                break;
        }

        gp.keyH.enterPressed = false;
    }

    public void trade_buy() {

        // DRAW PLAYER INVENTORY
        drawInventory(gp.player, false);
        // DRAW MERCHANT INVENTORY
        drawInventory(npc, true);

        // DRAW HINT WINDOW
        int x = gp.tileSize * 2;
        int y = gp.tileSize * 9;
        int width = gp.tileSize * 6;
        int height = gp.tileSize * 2;
        drawSubWindow(x, y, width, height);
        g2.drawString("[ESC] Back", x + 24, y + 60);

        // DRAW PLAYER COIN WINDOW
        x = gp.tileSize * 12;
        y = gp.tileSize * 9;
        width = gp.tileSize * 6;
        height = gp.tileSize * 2;
        drawSubWindow(x, y, width, height);
        g2.drawString("Your Coin: " + gp.player.coin, x + 24, y + 60);

        // DRAW PRICE WINDOW
        int itemIndex = getItemIdexOnSlot(npcSlotCol, npcSlotrow);
        if (itemIndex < npc.inventory.size()) {
            x = (int) (gp.tileSize * 5.5);
            y = (int) (gp.tileSize * 5.5);
            width = (int) (gp.tileSize * 2.5);
            height = gp.tileSize;
            drawSubWindow(x, y, width, height);
            g2.drawImage(coin, x + 10, y + 8, 32, 32, null);

            int price = npc.inventory.get(itemIndex).price;
            String text = "" + price;
            x = getXforAlignToRightText(text, gp.tileSize * 8 - 20);
            g2.drawString(text, x, y + 34);

            // BUY AN ITEM
            if (gp.keyH.enterPressed == true) {
                if (npc.inventory.get(itemIndex).price > gp.player.coin) {
                    subState = 0;
                    gp.gameState = gp.dialogueState;
                    currentDialogue = "You don't have enough resources!";
                    drawDialogueScreen();
                } else {
                    if (gp.player.canObtainItem(npc.inventory.get(itemIndex)) == true) {
                        gp.player.coin -= npc.inventory.get(itemIndex).price;
                    } else {
                        subState = 0;
                        gp.gameState = gp.dialogueState;
                        currentDialogue = "You can not carry any more items!";
                    }
                }
            }
        }
    }

    public void trade_sell() {
        drawInventory(gp.player, true);

        // DRAW HINT WINDOW
        int x = gp.tileSize * 2;
        int y = gp.tileSize * 9;
        int width = gp.tileSize * 6;
        int height = gp.tileSize * 2;
        drawSubWindow(x, y, width, height);
        g2.drawString("[ESC] Back", x + 24, y + 60);

        // DRAW PLAYER COIN WINDOW
        x = gp.tileSize * 12;
        y = gp.tileSize * 9;
        width = gp.tileSize * 6;
        height = gp.tileSize * 2;
        drawSubWindow(x, y, width, height);
        g2.drawString("Your Coin: " + gp.player.coin, x + 24, y + 60);

        // DRAW PRICE WINDOW
        int itemIndex = getItemIdexOnSlot(playerSlotCol, playerSlotrow);
        if (itemIndex < gp.player.inventory.size()) {
            x = (int) (gp.tileSize * 15.5);
            y = (int) (gp.tileSize * 5.5);
            width = (int) (gp.tileSize * 2.5);
            height = gp.tileSize;
            drawSubWindow(x, y, width, height);
            g2.drawImage(coin, x + 10, y + 8, 32, 32, null);

            int price = gp.player.inventory.get(itemIndex).price / 2;
            String text = "" + price;
            x = getXforAlignToRightText(text, gp.tileSize * 18 - 20);
            g2.drawString(text, x, y + 34);

            // SELL AN ITEM
            if (gp.keyH.enterPressed == true) {
                if (gp.player.inventory.get(itemIndex) == gp.player.currentWeapon
                        || gp.player.inventory.get(itemIndex) == gp.player.currentShield) {
                    commandNum = 0;
                    subState = 0;
                    gp.gameState = gp.dialogueState;
                    currentDialogue = "You can not sell an equiped item!";
                } else {
                    if (gp.player.inventory.get(itemIndex).amount > 1) {
                        gp.player.inventory.get(itemIndex).amount--;
                    } else {
                        gp.player.inventory.remove(itemIndex);
                    }
                    gp.player.coin += price;
                }
            }
        }
    }

    public void trade_select() {

        drawDialogueScreen();

        // DRAW WINDOW
        int x = gp.tileSize * 15;
        int y = gp.tileSize * 4;
        int width = gp.tileSize * 3;
        int height = (int) (gp.tileSize * 3.5);
        drawSubWindow(x, y, width, height);

        // TEXT
        x += gp.tileSize;
        y += gp.tileSize;
        g2.drawString("Buy", x, y);
        if (commandNum == 0) {
            g2.drawString(">", x - 24, y);
            if (gp.keyH.enterPressed == true) {
                subState = 1;
                gp.keyH.enterPressed = false;
                trade_buy();
            }
        }
        y += gp.tileSize;

        g2.drawString("Sell", x, y);
        if (commandNum == 1) {
            g2.drawString(">", x - 24, y);
            if (gp.keyH.enterPressed == true) {
                subState = 2;
                gp.keyH.enterPressed = false;
                trade_sell();
            }
        }
        y += gp.tileSize;

        g2.drawString("Leave", x, y);
        if (commandNum == 2) {
            g2.drawString(">", x - 24, y);
            if (gp.keyH.enterPressed == true) {
                commandNum = 0;
                gp.gameState = gp.dialogueState;
                currentDialogue = "Come again, hehe -><-";
            }
        }
        y += gp.tileSize;
    }

    public void drawSleepScreen() {

        counter++;

        if (counter < 120) {
            gp.eManager.lighting.filterAlpha += 0.01f;
            if (gp.eManager.lighting.filterAlpha > 1f) {
                gp.eManager.lighting.filterAlpha = 1f;
            }
        }
        if (counter >= 120) {
            gp.eManager.lighting.filterAlpha -= 0.01f;
            if (gp.eManager.lighting.filterAlpha <= 0f) {
                gp.eManager.lighting.filterAlpha = 0f;
                counter = 0;
                gp.eManager.lighting.dayState = gp.eManager.lighting.day;
                gp.eManager.lighting.dayCounter = 0;
                gp.gameState = gp.playState;
                gp.player.getImage();
            }
        }
    }

    public void drawSubWindow(int x, int y, int width, int height) {

        Color c = new Color(0, 0, 0, 210);
        g2.setColor(c);
        g2.fillRoundRect(x, y, width, height, 35, 35);

        c = new Color(255, 255, 255);
        g2.setColor(c);
        g2.setStroke(new BasicStroke(5));
        g2.drawRoundRect(x + 5, y + 5, width - 10, height - 10, 25, 25);

    }

    public int getXforCenteredText(String text) {

        int length = (int) g2.getFontMetrics().getStringBounds(text, g2).getWidth();
        int x = gp.screenWidth / 2 - length / 2;
        return x;
    }

    public int getXforAlignToRightText(String text, int tailX) {

        int length = (int) g2.getFontMetrics().getStringBounds(text, g2).getWidth();
        int x = tailX - length;
        return x;
    }
}