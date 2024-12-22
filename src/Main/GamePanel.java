package Main;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import javax.swing.JFrame;
import javax.swing.JPanel;

import ai.PathFinder;
import data.SaveLoad;
import entity.Entity;
import entity.Player;
import entity.Projectile;
import environment.EnvironmentManager;
import tile.TileManager;
import tile.Map;
import tile_interactive.InteractiveTile;

public class GamePanel extends JPanel implements Runnable {

    // SCREEN SETTINGS
    final int orignalTileSize = 16; // 16x16 tile
    final int scale = 3;

    public final int tileSize = orignalTileSize * scale; // 48x48 tile size
    public final int maxScreenColumn = 20;
    public final int maxScreenRow = 12;
    public final int screenWidth = tileSize * maxScreenColumn; // 960 pixels
    public final int screenHeight = tileSize * maxScreenRow; // 576 pixels

    // WORLD SETTINGS
    public int maxWorldColumn;
    public int maxWorldRow;

    public int worldWidth = tileSize * maxWorldColumn;
    public int worldHeight = tileSize * maxWorldRow;
    public final int maxMap = 10;
    public int currentMap = 0;

    // Full screen mode
    private int screenWidth2 = screenWidth;
    private int screenHeight2 = screenHeight;
    private BufferedImage tempScreen;// ON which the whole default screen is drawn as a image by using g2.draw
    private Graphics2D g2;
    boolean fullScreen = false;

    // FPS
    int FPS = 60;

    // SYSTEM
    public TileManager tileM = new TileManager(this);
    public KeyHandler keyH = new KeyHandler(this);
    Sound music = new Sound();
    Sound se = new Sound();
    public int sound = 0;
    public int soundEffects = 0;
    public CollisionChecker cChecker = new CollisionChecker(this);
    public AssetSetter aSetter = new AssetSetter(this);
    public UI ui = new UI(this);
    public EventHandler eHandler = new EventHandler(this);
    public PathFinder pFinder = new PathFinder(this);
    EnvironmentManager eManager = new EnvironmentManager(this);
    Map map = new Map(this);
    SaveLoad saveLoad = new SaveLoad(this);
    public EntityGenerator eGenerator = new EntityGenerator(this);
    Thread gameThread;

    // DEBUG FONT
    Font cambria = new Font("Cambria", Font.PLAIN, 32);

    // ENTITY AND OBJECT
    public Player player = new Player(this, keyH);
    public Entity obj[][] = new Entity[maxMap][25];
    public Entity npc[][] = new Entity[maxMap][10];
    public Entity monster[][] = new Entity[maxMap][20];
    public InteractiveTile iTile[][] = new InteractiveTile[maxMap][50];
    public Projectile projectile[][] = new Projectile[maxMap][30];
    public ArrayList<Entity> particleList = new ArrayList<>();
    ArrayList<Entity> entityList = new ArrayList<>();

    // GAME STATE
    public int gameState;
    public final int titleState = 0;
    public final int playState = 1;
    public final int pauseState = 2;
    public final int dialogueState = 3;
    public final int characterState = 4;
    public final int optionsState = 5;
    public final int gameOverState = 6;
    public final int transitionState = 7;
    public final int tradeState = 8;
    public final int sleepState = 9;
    public final int mapState = 10;

    public GamePanel() {
        this.setPreferredSize(new Dimension(screenWidth, screenHeight));
        this.setBackground(Color.BLACK);
        this.setDoubleBuffered(true);
        this.addKeyListener(keyH);
        this.setFocusable(true);

    }

    public void setUpGame() {

        aSetter.setObject();
        aSetter.setNPC();
        aSetter.setMonster();
        aSetter.setInteractiveTile();
        eManager.setup();

        gameState = titleState;

        tempScreen = new BufferedImage(screenWidth, screenHeight, BufferedImage.TYPE_INT_ARGB_PRE);
        g2 = (Graphics2D) tempScreen.getGraphics();

        setDefaultScreen();
    }

    public void resetGame(boolean restart) {

        System.out.println(player.speed);

        player.setDefaultPosition();
        player.restoreStatus();
        aSetter.setNPC();
        aSetter.setMonster();
        ui.message.clear();

        if (restart == true) {
            player.setDefaultValues();
            aSetter.setObject();
            aSetter.setInteractiveTile();
            eManager.lighting.resetDay();
        }
    }

    // public void retry() {

    // player.setDefaultPosition();
    // player.restoreLifeAndMana();
    // aSetter.setNPC();
    // aSetter.setMonster();
    // eManager.lighting.setDayState();
    // player.currentLight = null;
    // eManager.lighting.setLightSource();
    // }

    // public void restart() {
    // player.setDefaultValues();
    // player.setItems();
    // aSetter.setNPC();
    // aSetter.setObject();
    // aSetter.setInteractiveTile();
    // aSetter.setMonster();
    // eManager.lighting.setDayState();
    // player.currentLight = null;
    // eManager.lighting.setLightSource();

    // }

    public void setFullScreen() {

        // SET WINDOW TO FULL SCREEN MODE
        Main.window.setExtendedState(JFrame.MAXIMIZED_BOTH);

        // GET SCREEN DIMENSIONS
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        screenWidth2 = (int) screenSize.getWidth();
        screenHeight2 = (int) screenSize.getHeight();

        fullScreen = true;
        keyH.fPressed = false;
    }

    public void setDefaultScreen() {
        // Restore default window size
        Main.window.setExtendedState(JFrame.NORMAL);

        // GET DEFAULT SCREEN DIMENSIONS
        screenWidth2 = screenWidth;
        screenHeight2 = screenHeight;

        fullScreen = false;
        keyH.fPressed = false;
    }

    public void startGameThread() {
        gameThread = new Thread(this);
        gameThread.start();
    }

    public void run() {

        double drawInterval = 1000000000 / FPS;
        double delta = 0;
        long lastTime = System.nanoTime();
        long currentTime;
        long timer = 0;
        int drawCount = 0;

        while (gameThread != null) {
            currentTime = System.nanoTime();
            delta += (currentTime - lastTime) / drawInterval;
            timer += currentTime - lastTime;
            lastTime = currentTime;

            if (delta > 1) {

                update();

                // FOR FULL SCREEN
                drawToTempScreen(); // draw everything to buffered image
                drawToScreen(); // draw the buffered image to the screen
                if (keyH.fPressed == true) {
                    if (fullScreen == false) {
                        setFullScreen();
                    } else {
                        setDefaultScreen();
                    }

                }
                delta--;
                drawCount++;
            }

            if (timer >= 1000000000) {
                System.out.println("FPS: " + drawCount);
                drawCount = 0;
                timer = 0;
            }
        }
    }

    public void update() {
        if (gameState == playState) {

            // PLAYER
            player.update();

            // NPC
            for (int i = 0; i < npc[1].length; i++) {
                if (npc[currentMap][i] != null) {
                    npc[currentMap][i].update();
                }
            }

            // MONSTER
            for (int i = 0; i < monster[1].length; i++) {
                if (monster[currentMap][i] != null) {
                    if (monster[currentMap][i].alive == true && monster[currentMap][i].dying == false) {
                        monster[currentMap][i].update();
                    }
                    if (monster[currentMap][i].alive == false) {
                        monster[currentMap][i].checkDrop();
                        monster[currentMap][i] = null;
                    }
                }
            }

            // PROJECTLE
            for (int i = 0; i < projectile[1].length; i++) {
                if (projectile[currentMap][i] != null) {
                    if (projectile[currentMap][i].alive == true) {
                        projectile[currentMap][i].update();
                    }
                    if (projectile[currentMap][i].alive == false) {
                        projectile[currentMap][i] = null;
                    }
                }
            }

            // PARTICLE
            for (int i = 0; i < particleList.size(); i++) {
                if (particleList.get(i) != null) {
                    if (particleList.get(i).alive == true) {
                        particleList.get(i).update();
                    }
                    if (particleList.get(i).alive == false) {
                        particleList.remove(i);
                    }
                }
            }

            // INTERACTIVE TILE
            for (int i = 0; i < iTile[1].length; i++) {
                if (iTile[currentMap][i] != null) {
                    iTile[currentMap][i].update();
                }
            }
            eManager.update();

        }
        if (gameState == pauseState) {
            // nothing
        }
    }

    public void drawToTempScreen() {

        g2.setColor(Color.BLACK);
        g2.fillRect(0, 0, screenWidth, screenHeight);

        // DEBUG
        long drawStart = 0;
        if (keyH.showDebugText == true) {
            drawStart = System.nanoTime();

        }
        // TITLE SCREEN
        if (gameState == titleState) {
            ui.draw(g2);
        }
        // MAP SCREEN
        else if (gameState == mapState) {
            map.drawFullMapScreen(g2);
        }
        // OTHERS
        else {

            // TILE
            tileM.draw(g2);

            // INTERACTIVE TILE
            for (int i = 0; i < iTile[1].length; i++) {
                if (iTile[currentMap][i] != null) {
                    iTile[currentMap][i].draw(g2);
                }
            }

            // ADD ENTITIES TO THE LIST
            entityList.add(player);

            for (int i = 0; i < npc[1].length; i++) {
                if (npc[currentMap][i] != null) {
                    entityList.add(npc[currentMap][i]);
                }
            }

            for (int i = 0; i < obj[1].length; i++) {
                if (obj[currentMap][i] != null) {
                    // System.out.println(i);
                    // System.out.println("mapNum: " + currentMap);
                    // System.out.println("name: " + obj[currentMap][i].name);
                    // System.out.println("worldX: " + obj[currentMap][i].worldX);
                    // System.out.println("worldY: " + obj[currentMap][i].worldY);
                    // System.out.println("loot: " + obj[currentMap][i].loot);
                    // System.out.println("opened: " + obj[currentMap][i].opened);
                    // if ((i % 2) == 0) {
                    // System.out.println("--------");

                    // }
                    // System.out.println("------------------------");
                    entityList.add(obj[currentMap][i]);
                }
            }

            for (int i = 0; i < monster[1].length; i++) {
                if (monster[currentMap][i] != null) {
                    entityList.add(monster[currentMap][i]);
                }
            }

            for (int i = 0; i < projectile[1].length; i++) {
                if (projectile[currentMap][i] != null) {
                    entityList.add(projectile[currentMap][i]);
                }
            }

            for (int i = 0; i < particleList.size(); i++) {
                if (particleList.get(i) != null) {
                    entityList.add(particleList.get(i));
                }
            }

            // SORT
            Collections.sort(entityList, new Comparator<Entity>() {

                @Override
                public int compare(Entity e1, Entity e2) {

                    int result = Integer.compare(e1.worldY, e2.worldY);
                    return result;
                }

            });

            // DRAW ENTITIES
            for (int i = 0; i < entityList.size(); i++) {
                entityList.get(i).draw(g2);
            }

            // EMPTY LIST
            entityList.clear();

            // ENVIRONMENT
            eManager.draw(g2);

            // MINI MAP
            map.drawMiniMap(g2);

            // UI
            ui.draw(g2);

            // DEBUG
            if (keyH.showDebugText == true) {
                long drawEnd = System.nanoTime();
                long passed = drawEnd - drawStart;

                g2.setFont(new Font("Arial", Font.PLAIN, 20));
                g2.setColor(Color.white);
                int x = 10;
                int y = 400;
                int lineHeight = 20;

                g2.drawString("WorldX: " + player.worldX, x, y);
                y += lineHeight;
                g2.drawString("WorldY: " + player.worldY, x, y);
                y += lineHeight;
                g2.drawString("Col: " + (player.worldX + player.solidArea.x) / tileSize, x, y);
                y += lineHeight;
                g2.drawString("Row: " + (player.worldY + player.solidArea.y) / tileSize, x, y);
                y += lineHeight;

                g2.drawString("Draw Time: " + passed, x, y);
            }

        }

    }

    public void drawToScreen() {
        Graphics g = this.getGraphics();

        // Check if graphics context is available before drawing
        if (g != null) {
            g.drawImage(tempScreen, 0, 0, screenWidth2, screenHeight2, null);
            g.dispose();
        }
    }

    public void playMusic(int i) {

        music.setFile(i);
        music.play();
        music.loop();
    }

    public void stopMusic() {
        music.stop();
    }

    public void playSE(int i) {

        se.setFile(i);
        se.play();
    }
}
