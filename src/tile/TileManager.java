package tile;

import java.awt.Color;
import java.awt.Graphics2D;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import Main.GamePanel;
import Main.UtilityTool;

public class TileManager {
    GamePanel gp;
    public Tile[] tile;
    public int mapTileNum[][][];
    boolean drawPath = false;
    ArrayList<String> fileNames = new ArrayList<>();
    ArrayList<String> collisionStatus = new ArrayList<>();

    public TileManager(GamePanel gp) {
        this.gp = gp;

        // READ TILE DATA FILE
        InputStream is = getClass().getResourceAsStream("/maps/tiledata.txt");
        BufferedReader br = new BufferedReader(new InputStreamReader(is));

        // GETTING TILE NAMES AND COLLISION INFO FROM THE FILE
        String line;

        try {
            while ((line = br.readLine()) != null) {
                fileNames.add(line);
                collisionStatus.add(br.readLine());
            }
            br.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

        tile = new Tile[fileNames.size()];
        getTileImage();

        // get maxWorldCol & row
        is = getClass().getResourceAsStream("/maps/worldmap.txt");
        br = new BufferedReader(new InputStreamReader(is));

        try {
            String line2 = br.readLine();
            String maxTile[] = line2.split(" ");

            gp.maxWorldColumn = maxTile.length;
            gp.maxWorldRow = maxTile.length;

            gp.worldWidth = gp.tileSize * gp.maxWorldColumn;
            gp.worldHeight = gp.tileSize * gp.maxWorldRow;

            mapTileNum = new int[gp.maxMap][gp.maxWorldColumn][gp.maxWorldRow];

            br.close();

        } catch (IOException e) {
            System.out.println("Exception!");
        }

        loadMap("/maps/worldmap.txt", 0);

        // loadMap("/maps/worldV3.txt", 0);
        loadMap("/maps/indoor01.txt", 1);
    }

    public void getTileImage() {

        for (int i = 0; i < fileNames.size(); i++) {

            String fileName;
            boolean collision;

            // Get a file name
            fileName = fileNames.get(i);

            // Get collision status
            if (collisionStatus.get(i).equals("true")) {
                collision = true;
            } else {
                collision = false;
            }

            setup(i, fileName, collision);
        }
    }

    public void setup(int index, String imageName, boolean collision) {

        UtilityTool uTool = new UtilityTool();

        try {
            tile[index] = new Tile();
            tile[index].image = ImageIO.read(getClass().getResourceAsStream("/tiles/" + imageName));
            tile[index].image = uTool.scaleImage(tile[index].image, gp.tileSize, gp.tileSize);
            tile[index].collision = collision;

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void loadMap(String filePath, int map) {
        try {
            InputStream is = getClass().getResourceAsStream(filePath);
            BufferedReader br = new BufferedReader(new InputStreamReader(is));

            int col = 0;
            int row = 0;

            while (col < gp.maxWorldColumn && row < gp.maxWorldRow) {
                String line = br.readLine();

                while (col < gp.maxWorldColumn) {

                    String numbers[] = line.split(" ");

                    int num = Integer.parseInt(numbers[col]);

                    mapTileNum[map][col][row] = num;
                    col++;

                }
                if (col == gp.maxWorldColumn) {

                    col = 0;
                    row++;
                }

            }
            br.close();
        } catch (Exception e) {

        }
    }

    public void draw(Graphics2D g2) {
        int worldColumn = 0;
        int worldRow = 0;

        while (worldColumn < gp.maxWorldColumn && worldRow < gp.maxWorldRow) {

            int tileNum = mapTileNum[gp.currentMap][worldColumn][worldRow];

            int worldX = worldColumn * gp.tileSize;
            int worldY = worldRow * gp.tileSize;
            int screenX = worldX - gp.player.worldX + gp.player.screenX;
            int screenY = worldY - gp.player.worldY + gp.player.screenY;

            // Stop moving camera at edge
            // only draw tiles that are shown
            if (gp.player.screenX > gp.player.worldX) {
                screenX = worldX;
            }
            if (gp.player.screenY > gp.player.worldY) {
                screenY = worldY;
            }
            int rightOffSet = gp.screenWidth - gp.player.screenX;
            if (rightOffSet > gp.worldWidth - gp.player.worldX) {
                screenX = gp.screenWidth - (gp.worldWidth - worldX);
            }
            int bottomOffSet = gp.screenHeight - gp.player.screenY;
            if (bottomOffSet > gp.worldHeight - gp.player.worldY) {
                screenY = gp.screenHeight - (gp.worldHeight - worldY);
            }
            //////////

            if (worldX + gp.tileSize > gp.player.worldX - gp.player.screenX &&
                    worldX - gp.tileSize < gp.player.worldX + gp.player.screenX &&
                    worldY + gp.tileSize > gp.player.worldY - gp.player.screenY &&
                    worldY - gp.tileSize < gp.player.worldY + gp.player.screenY) {

                g2.drawImage(tile[tileNum].image, screenX, screenY, null);

            } else if (gp.player.screenX > gp.player.worldX ||
                    gp.player.screenY > gp.player.worldY ||
                    rightOffSet > gp.worldWidth - gp.player.worldX ||
                    bottomOffSet > gp.worldHeight - gp.player.worldY) {
                g2.drawImage(tile[tileNum].image, screenX, screenY, null);
            }
            worldColumn++;

            if (worldColumn == gp.maxWorldColumn) {
                worldColumn = 0;
                worldRow++;
            }

        }

        if (drawPath == true) {
            g2.setColor(new Color(255, 0, 0, 70));

            for (int i = 0; i < gp.pFinder.pathList.size(); i++) {

                int worldX = gp.pFinder.pathList.get(i).col * gp.tileSize;
                int worldY = gp.pFinder.pathList.get(i).row * gp.tileSize;
                int screenX = worldX - gp.player.worldX + gp.player.screenX;
                int screenY = worldY - gp.player.worldY + gp.player.screenY;

                g2.fillRect(screenX, screenY, gp.tileSize, gp.tileSize);

            }
        }

    }

}
