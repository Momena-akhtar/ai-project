package Main;

import entity.Entity;

public class CollisionChecker {
    GamePanel gp;

    public CollisionChecker(GamePanel gp) {
        this.gp = gp;
    }

    public void checkTile(Entity entity) {

        int entityLeftWorldX = entity.worldX + entity.solidArea.x;
        int entityRightWorldX = entity.worldX + entity.solidArea.x + entity.solidArea.width;
        int entityTopWorldY = entity.worldY + entity.solidArea.y;
        int entityBottomWorldY = entity.worldY + entity.solidArea.y + entity.solidArea.height;

        int entityLeftColumn = entityLeftWorldX / gp.tileSize;
        int entityRightColumn = entityRightWorldX / gp.tileSize;
        int entityTopRow = entityTopWorldY / gp.tileSize;
        int entityBottomRow = entityBottomWorldY / gp.tileSize;

        int tileNum1, tileNum2, tileNum3, tileNum4;

        // Use tempporal direction when it's being knockbacked
        String direction = entity.direction;
        if (entity.knockBack == true) {
            direction = entity.knockBackDirection;
        }

        switch (direction) {
            case "up":
                entityTopRow = (entityTopWorldY - entity.speed) / gp.tileSize;
                tileNum1 = gp.tileM.mapTileNum[gp.currentMap][entityLeftColumn][entityTopRow];
                tileNum2 = gp.tileM.mapTileNum[gp.currentMap][entityRightColumn][entityTopRow];

                if (gp.tileM.tile[tileNum1].collision == true || gp.tileM.tile[tileNum2].collision == true) {
                    entity.collisionOn = true;
                }

                break;
            case "down":
                entityBottomRow = (entityBottomWorldY + entity.speed) / gp.tileSize;
                tileNum3 = gp.tileM.mapTileNum[gp.currentMap][entityLeftColumn][entityBottomRow];
                tileNum4 = gp.tileM.mapTileNum[gp.currentMap][entityRightColumn][entityBottomRow];

                if (gp.tileM.tile[tileNum3].collision == true || gp.tileM.tile[tileNum4].collision == true) {
                    entity.collisionOn = true;
                }

                break;
            case "left":
                entityLeftColumn = (entityLeftWorldX - entity.speed) / gp.tileSize;
                tileNum1 = gp.tileM.mapTileNum[gp.currentMap][entityLeftColumn][entityTopRow];
                tileNum3 = gp.tileM.mapTileNum[gp.currentMap][entityLeftColumn][entityBottomRow];

                if (gp.tileM.tile[tileNum1].collision == true || gp.tileM.tile[tileNum3].collision == true) {
                    entity.collisionOn = true;
                }

                break;
            case "right":
                entityRightColumn = (entityRightWorldX + entity.speed) / gp.tileSize;
                tileNum2 = gp.tileM.mapTileNum[gp.currentMap][entityRightColumn][entityTopRow];
                tileNum4 = gp.tileM.mapTileNum[gp.currentMap][entityRightColumn][entityBottomRow];

                if (gp.tileM.tile[tileNum2].collision == true || gp.tileM.tile[tileNum4].collision == true) {
                    entity.collisionOn = true;
                }

                break;

            default:
                break;
        }

    }

    public int checkObject(Entity entity, boolean player) {

        int index = 999;

        // Use tempporal direction when it's being knockbacked
        String direction = entity.direction;
        if (entity.knockBack == true) {
            direction = entity.knockBackDirection;
        }

        for (int i = 0; i < gp.obj[1].length; i++) {

            if (gp.obj[gp.currentMap][i] != null) {

                // get entity's solid area position
                entity.solidArea.x = entity.worldX + entity.solidArea.x;
                entity.solidArea.y = entity.worldY + entity.solidArea.y;

                // get object's solid area position
                gp.obj[gp.currentMap][i].solidArea.x = gp.obj[gp.currentMap][i].worldX
                        + gp.obj[gp.currentMap][i].solidArea.x;
                gp.obj[gp.currentMap][i].solidArea.y = gp.obj[gp.currentMap][i].worldY
                        + gp.obj[gp.currentMap][i].solidArea.y;
                // The last part is here to change the dimensions later if you want to, rn it
                // does not have any effect

                switch (direction) {
                    case "up":
                        entity.solidArea.y -= entity.speed;
                        break;
                    case "down":
                        entity.solidArea.y += entity.speed;
                        break;
                    case "left":
                        entity.solidArea.x -= entity.speed;
                        break;
                    case "right":
                        entity.solidArea.x += entity.speed;
                        break;
                }
                if (entity.solidArea.intersects(gp.obj[gp.currentMap][i].solidArea)) {
                    if (gp.obj[gp.currentMap][i].collision == true) {
                        entity.collisionOn = true;
                    }

                    if (player == true) {
                        index = i;
                    }
                }

                entity.solidArea.x = entity.solidAreaDefaultX;
                entity.solidArea.y = entity.solidAreaDefaultY;
                gp.obj[gp.currentMap][i].solidArea.x = gp.obj[gp.currentMap][i].solidAreaDefaultX;
                gp.obj[gp.currentMap][i].solidArea.y = gp.obj[gp.currentMap][i].solidAreaDefaultY;

            }
        }

        return index;

    }

    // NPC OR MONSTER COLLISION
    public int checkEntity(Entity entity, Entity[][] target) {

        int index = 999;

        // Use tempporal direction when it's being knockbacked
        String direction = entity.direction;
        if (entity.knockBack == true) {
            direction = entity.knockBackDirection;
        }

        for (int i = 0; i < target[1].length; i++) {

            if (target[gp.currentMap][i] != null) {

                // get entity's solid area position
                entity.solidArea.x = entity.worldX + entity.solidArea.x;
                entity.solidArea.y = entity.worldY + entity.solidArea.y;

                // get object's solid area position
                target[gp.currentMap][i].solidArea.x = target[gp.currentMap][i].worldX
                        + target[gp.currentMap][i].solidArea.x;
                target[gp.currentMap][i].solidArea.y = target[gp.currentMap][i].worldY
                        + target[gp.currentMap][i].solidArea.y;
                // The last part is here to change the dimensions later if you want to, rn it
                // does not have any effect

                switch (direction) {
                    case "up":
                        entity.solidArea.y -= entity.speed;
                        break;
                    case "down":
                        entity.solidArea.y += entity.speed;
                        break;
                    case "left":
                        entity.solidArea.x -= entity.speed;
                        break;
                    case "right":
                        entity.solidArea.x += entity.speed;
                        break;
                }
                if (entity.solidArea.intersects(target[gp.currentMap][i].solidArea)) {
                    if (target[gp.currentMap][i] != entity) {
                        entity.collisionOn = true;
                        index = i;
                    }
                }
                entity.solidArea.x = entity.solidAreaDefaultX;
                entity.solidArea.y = entity.solidAreaDefaultY;
                target[gp.currentMap][i].solidArea.x = target[gp.currentMap][i].solidAreaDefaultX;
                target[gp.currentMap][i].solidArea.y = target[gp.currentMap][i].solidAreaDefaultY;

            }
        }

        return index;

    }

    public boolean checkPlayer(Entity entity) {

        boolean contactPlayer = false;

        // get entity's solid area position
        entity.solidArea.x = entity.worldX + entity.solidArea.x;
        entity.solidArea.y = entity.worldY + entity.solidArea.y;

        // get object's solid area position
        gp.player.solidArea.x = gp.player.worldX + gp.player.solidArea.x;
        gp.player.solidArea.y = gp.player.worldY + gp.player.solidArea.y;
        // The last part is here to change the dimensions later if you want to, rn it
        // does not have any effect

        switch (entity.direction) {
            case "up":
                entity.solidArea.y -= entity.speed;
                break;
            case "down":
                entity.solidArea.y += entity.speed;
                break;
            case "left":
                entity.solidArea.x -= entity.speed;
                break;
            case "right":
                entity.solidArea.x += entity.speed;
                break;
        }
        if (entity.solidArea.intersects(gp.player.solidArea)) {
            entity.collisionOn = true;
            contactPlayer = true;
        }

        entity.solidArea.x = entity.solidAreaDefaultX;
        entity.solidArea.y = entity.solidAreaDefaultY;
        gp.player.solidArea.x = gp.player.solidAreaDefaultX;
        gp.player.solidArea.y = gp.player.solidAreaDefaultY;

        return contactPlayer;
    }

}
