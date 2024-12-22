package Main;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class KeyHandler implements KeyListener {

    GamePanel gp;

    public boolean upPressed, downPressed, rightPressed, leftPressed, enterPressed, spacePressed, fPressed, bPressed;

    // DEBUG
    boolean showDebugText = false;

    public KeyHandler(GamePanel gp) {
        this.gp = gp;
    }

    @Override
    public void keyTyped(KeyEvent e) {
    }

    @Override
    public void keyPressed(KeyEvent e) {
        int code = e.getKeyCode();

        if (code == KeyEvent.VK_F) {
            if (code == KeyEvent.VK_F) {
                fPressed = true;
            }
        }

        // TITLE STATE
        if (gp.gameState == gp.titleState) {
            titleState(code);
        }
        // PLAY STATE
        else if (gp.gameState == gp.playState) {
            playState(code);

            // DEBUG
            if (code == KeyEvent.VK_T) {
                if (showDebugText == false) {
                    showDebugText = true;
                } else if (showDebugText == true) {
                    showDebugText = false;
                }
            }
            if (code == KeyEvent.VK_R) {
                switch (gp.currentMap) {
                    case 0:
                        gp.tileM.loadMap("/maps/worldV3.txt", 0);
                        break;
                    case 1:
                        gp.tileM.loadMap("/maps/interior01.txt", 1);
                        break;
                }
            }
        }
        // PAUSE STATE
        else if (gp.gameState == gp.pauseState) {
            pauseState(code);
        }
        // DIALOGUE STATE
        else if (gp.gameState == gp.dialogueState) {
            dialogueState(code);
        }
        // CHARACTER STATE
        else if (gp.gameState == gp.characterState) {
            characterState(code);
        }
        // OPTIONS STATE
        else if (gp.gameState == gp.optionsState) {
            optionsState(code);
        }
        // GAME OVER STATE
        else if (gp.gameState == gp.gameOverState) {
            gameOverState(code);
        }
        // TRADE STATE
        else if (gp.gameState == gp.tradeState) {
            tradeState(code);
        }
        // MAP STATE
        else if (gp.gameState == gp.mapState) {
            mapState(code);
        }
    }

    public void tradeState(int code) {

        if (code == KeyEvent.VK_ENTER) {
            enterPressed = true;
        }

        if (gp.ui.subState == 0) {

            if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
                gp.ui.commandNum--;
                gp.playSE(10);
                if (gp.ui.commandNum < 0) {
                    gp.ui.commandNum = 2;
                }
            }
            if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
                gp.ui.commandNum++;
                gp.playSE(10);
                if (gp.ui.commandNum > 2) {
                    gp.ui.commandNum = 0;
                }
            }
        }
        if (gp.ui.subState == 1) {
            npcInventory(code);
            if (code == KeyEvent.VK_ESCAPE) {
                gp.ui.subState = 0;
            }
        }
        if (gp.ui.subState == 2) {
            playerInventory(code);
            if (code == KeyEvent.VK_ESCAPE) {
                gp.ui.subState = 0;
            }
        }
    }

    public void optionsState(int code) {
        if (code == KeyEvent.VK_ESCAPE) {
            gp.gameState = gp.playState;
            gp.ui.commandNum = 0;
        }
        if (code == KeyEvent.VK_ENTER) {
            enterPressed = true;
        }

        int maxCommandNum = 0;
        switch (gp.ui.subState) {
            case 0:
                maxCommandNum = 5;
                break;
            case 1:
                maxCommandNum = 0;
                break;
            case 2:
                maxCommandNum = 1;
                break;
            case 3:
                maxCommandNum = 5;
                break;
        }

        if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
            gp.ui.commandNum--;
            gp.playSE(10);
            if (gp.ui.commandNum < 0) {
                gp.ui.commandNum = maxCommandNum;
            }
        }
        if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
            gp.ui.commandNum++;
            gp.playSE(10);
            if (gp.ui.commandNum > maxCommandNum) {
                gp.ui.commandNum = 0;
            }
        }
        if (code == KeyEvent.VK_A || code == KeyEvent.VK_LEFT) {
            if (gp.ui.subState == 0) {
                if (gp.ui.commandNum == 1 && gp.music.volumeScale > 0) {
                    gp.music.volumeScale--;
                    gp.music.checkVolume();
                    gp.playSE(10);
                }
                if (gp.ui.commandNum == 2 && gp.se.volumeScale > 0) { // Adjusted bounds check
                    gp.se.volumeScale--;
                    if (gp.se.volumeScale < 0) {
                        gp.se.volumeScale = 0;
                    }
                    gp.playSE(10);
                }
            }
        }
        if (code == KeyEvent.VK_D || code == KeyEvent.VK_RIGHT) {
            if (gp.ui.subState == 0) {
                if (gp.ui.commandNum == 1 && gp.music.volumeScale < 5) {
                    gp.music.volumeScale++;
                    gp.music.checkVolume();
                    gp.playSE(10);
                }
                if (gp.ui.commandNum == 2 && gp.se.volumeScale < 5) { // Adjusted bounds check
                    gp.se.volumeScale++;
                    if (gp.se.volumeScale > 5) {
                        gp.se.volumeScale = 5;
                    }
                    gp.playSE(10);
                }
            }
        }

    }

    public void titleState(int code) {

        if (gp.ui.titleScreenState == 0) {

            if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
                gp.ui.commandNum--;
                if (gp.ui.commandNum < 0) {
                    gp.ui.commandNum = 2;
                }
            }
            if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {

                gp.ui.commandNum++;
                if (gp.ui.commandNum > 2) {
                    gp.ui.commandNum = 0;
                }
            }
            if (code == KeyEvent.VK_ENTER) {
                switch (gp.ui.commandNum) {
                    case 0:
                        gp.ui.titleScreenState = 1;
                        gp.ui.commandNum = 0;
                        break;

                    case 1:
                        gp.saveLoad.load();
                        gp.ui.titleScreenState = 1;
                        gp.ui.commandNum = 0;
                        break;

                    case 2:
                        System.exit(0);
                        break;
                }
            }
        } else if (gp.ui.titleScreenState == 1) {
            if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
                gp.ui.commandNum--;
                if (gp.ui.commandNum < 0) {
                    gp.ui.commandNum = 3;
                }
            }
            if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
                gp.ui.commandNum++;
                if (gp.ui.commandNum > 3) {
                    gp.ui.commandNum = 0;
                }
            }
            if (code == KeyEvent.VK_ENTER) {
                switch (gp.ui.commandNum) {
                    case 0:
                        System.out.println("do some fighter specific stuff");
                        gp.ui.titleScreenState = 0;
                        gp.gameState = gp.playState;
                        gp.playMusic(gp.sound);
                        break;

                    case 1:
                        System.out.println("do some thief specific stuff");
                        gp.ui.titleScreenState = 0;
                        gp.gameState = gp.playState;
                        gp.playMusic(gp.sound);
                        break;

                    case 2:
                        System.out.println("do some sorcerer specific stuff");
                        gp.ui.titleScreenState = 0;
                        gp.gameState = gp.playState;
                        gp.playMusic(gp.sound);
                        break;

                    case 3:
                        gp.ui.titleScreenState = 0;
                        gp.ui.commandNum = 0;
                        break;
                }
            }
        }
    }

    public void playState(int code) {

        if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
            upPressed = true;
        }
        if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
            downPressed = true;
        }
        if (code == KeyEvent.VK_A || code == KeyEvent.VK_LEFT) {
            leftPressed = true;
        }
        if (code == KeyEvent.VK_D || code == KeyEvent.VK_RIGHT) {
            rightPressed = true;
        }
        if (code == KeyEvent.VK_P) {
            gp.gameState = gp.pauseState;
        }
        if (code == KeyEvent.VK_C) {
            gp.gameState = gp.characterState;
        }
        if (code == KeyEvent.VK_ENTER) {
            enterPressed = true;
        }
        if (code == KeyEvent.VK_SPACE) {
            spacePressed = true;
        }
        if (code == KeyEvent.VK_ESCAPE) {
            gp.gameState = gp.optionsState;
        }
        if (code == KeyEvent.VK_M) {
            gp.gameState = gp.mapState;
        }
        if (code == KeyEvent.VK_B) {
            bPressed = true;
        }
        if (code == KeyEvent.VK_X) {
            if (gp.map.miniMapOn == false) {
                gp.map.miniMapOn = true;
            } else {
                gp.map.miniMapOn = false;
            }
        }
    }

    public void pauseState(int code) {

        if (code == KeyEvent.VK_P) {
            gp.gameState = gp.playState;

        }
    }

    public void dialogueState(int code) {

        if (code == KeyEvent.VK_ENTER) {
            gp.gameState = gp.playState;
        }
    }

    public void characterState(int code) {

        if (code == KeyEvent.VK_C) {
            gp.gameState = gp.playState;
            gp.playSE(10);
        }
        if (code == KeyEvent.VK_ENTER) {
            gp.player.selectItem();
        }
        playerInventory(code);
    }

    public void gameOverState(int code) {

        if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
            gp.ui.commandNum--;
            gp.playSE(10);
            if (gp.ui.commandNum < 0) {
                gp.ui.commandNum = 1;
            }
        }
        if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
            gp.ui.commandNum++;
            gp.playSE(10);
            if (gp.ui.commandNum > 1) {
                gp.ui.commandNum = 0;
            }
        }

        if (code == KeyEvent.VK_ENTER) {
            if (gp.ui.commandNum == 0) {
                gp.gameState = gp.playState;
                gp.resetGame(false);
                gp.playMusic(0);

            } else if (gp.ui.commandNum == 1) {
                gp.gameState = gp.titleState;
                gp.ui.commandNum = 0;
                gp.resetGame(true);
            }
        }
    }

    public void playerInventory(int code) {
        if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
            if (gp.ui.playerSlotrow != 0) {
                gp.ui.playerSlotrow--;
                gp.playSE(10);
            }
        }
        if (code == KeyEvent.VK_A || code == KeyEvent.VK_LEFT) {
            if (gp.ui.playerSlotCol != 0) {
                gp.ui.playerSlotCol--;
                gp.playSE(10);
            }
        }
        if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
            if (gp.ui.playerSlotrow != 3) {
                gp.ui.playerSlotrow++;
                gp.playSE(10);
            }
        }
        if (code == KeyEvent.VK_D || code == KeyEvent.VK_RIGHT) {
            if (gp.ui.playerSlotCol != 4) {
                gp.ui.playerSlotCol++;
                gp.playSE(10);
            }
        }
    }

    public void mapState(int code) {

        if (code == KeyEvent.VK_M) {
            gp.gameState = gp.playState;
        }
    }

    public void npcInventory(int code) {
        if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
            if (gp.ui.npcSlotrow != 0) {
                gp.ui.npcSlotrow--;
                gp.playSE(10);
            }
        }
        if (code == KeyEvent.VK_A || code == KeyEvent.VK_LEFT) {
            if (gp.ui.npcSlotCol != 0) {
                gp.ui.npcSlotCol--;
                gp.playSE(10);
            }
        }
        if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
            if (gp.ui.npcSlotrow != 3) {
                gp.ui.npcSlotrow++;
                gp.playSE(10);
            }
        }
        if (code == KeyEvent.VK_D || code == KeyEvent.VK_RIGHT) {
            if (gp.ui.npcSlotCol != 4) {
                gp.ui.npcSlotCol++;
                gp.playSE(10);
            }
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {
        int code = e.getKeyCode();

        if (code == KeyEvent.VK_W || code == KeyEvent.VK_UP) {
            upPressed = false;
        }
        if (code == KeyEvent.VK_S || code == KeyEvent.VK_DOWN) {
            downPressed = false;
        }
        if (code == KeyEvent.VK_A || code == KeyEvent.VK_LEFT) {
            leftPressed = false;
        }
        if (code == KeyEvent.VK_D || code == KeyEvent.VK_RIGHT) {
            rightPressed = false;
        }
        if (code == KeyEvent.VK_SPACE) {
            spacePressed = false;
        }
        if (code == KeyEvent.VK_B) {
            bPressed = false;
        }
        if (code == KeyEvent.VK_ENTER) {
            enterPressed = false;
        }
    }

}
